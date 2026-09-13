"""Persistent RK3576 worker for Audio8 deployment buckets.

The worker deliberately resets prefill and external KV for every task.  It
keeps RKNN objects resident, which removes per-segment model-load overhead.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


PROFILES = {
    "s8": ("encoder/audio_encoder_f800_fp16.rknn", "adapter/audio_adapter_h104_t100_fp16.rknn", "prefill/prefill_kv_s110_fp16.rknn"),
    "s16": ("encoder/audio_encoder_f1600_fp16.rknn", "adapter/audio_adapter_h208_t200_fp16.rknn", "prefill/prefill_kv_s210_fp16.rknn"),
    "s30": ("encoder/audio_encoder_f3000_fp16.rknn", "adapter/audio_adapter_h390_t375_fp16.rknn", "prefill/prefill_kv_s385_fp16.rknn"),
}


def load(path: Path) -> RKNNLite:
    if not path.is_file():
        raise FileNotFoundError(f"missing RKNN asset: {path}")
    model = RKNNLite(verbose=False)
    if model.load_rknn(str(path)) != 0 or model.init_runtime() != 0:
        raise RuntimeError(f"cannot initialize {path}")
    return model


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


class Worker:
    def __init__(self, root: Path, embeddings: Path, capacities: tuple[int, ...]) -> None:
        self.root, self.table, self.capacities = root, np.load(embeddings, mmap_mode="r"), capacities
        # A 4 GiB RK3576 cannot keep the large frontend graphs and all decoder
        # graphs mapped at the same time.  Keep the process alive but enforce a
        # stage lifetime: frontend → release → decoder.  This is intentional,
        # not a fallback to CPU.
        self.frontends: dict[str, tuple[RKNNLite, RKNNLite, RKNNLite]] = {}
        self.kvs: list[RKNNLite] = []
        self.heads: list[RKNNLite] = []
        self.blocks: dict[int, list[RKNNLite]] = {}

    def frontend(self, profile: str) -> tuple[RKNNLite, RKNNLite, RKNNLite]:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile}; expected one of {sorted(PROFILES)}")
        if profile not in self.frontends:
            self.frontends[profile] = tuple(load(self.root / item) for item in PROFILES[profile])
        return self.frontends[profile]

    def block(self, capacity: int) -> list[RKNNLite]:
        if capacity not in self.blocks:
            self.blocks[capacity] = [load(self.root / f"decoder/block_s{capacity}/layer{i}/block_fp16.rknn") for i in range(8)]
        return self.blocks[capacity]

    def decoder(self) -> tuple[list[RKNNLite], list[RKNNLite]]:
        if not self.kvs:
            self.kvs = [load(self.root / f"decoder/kv/layer{i}/kv_fp16.rknn") for i in range(8)]
            self.heads = [load(self.root / f"decoder/head_shards/shard{i:02d}/head_fp16.rknn") for i in range(8)]
        return self.kvs, self.heads

    def release_frontends(self) -> None:
        for group in self.frontends.values():
            for item in group:
                item.release()
        self.frontends.clear()

    def release_decoder(self) -> None:
        for group in [self.kvs, self.heads, *self.blocks.values()]:
            for item in group:
                item.release()
        self.kvs, self.heads, self.blocks = [], [], {}

    def run(self, task: dict[str, object], max_new: int, eos: set[int]) -> dict[str, object]:
        started = time.perf_counter()
        # The preceding task leaves only decoder models resident.  Unmap them
        # before mapping this task's frontend profile; the two stages cannot
        # coexist within the board's memory budget.
        self.release_decoder()
        input_dir = Path(str(task["input_dir"]))
        enc, adapter, prefill = self.frontend(str(task["profile"]))
        frontend_rss = rss_mib()
        mel = np.load(input_dir / "input_features.npy").astype("float32")
        ids = np.load(input_dir / "input_ids.npy").reshape(-1)
        audio_positions = np.load(input_dir / "audio_positions.npy").reshape(-1)
        t = time.perf_counter(); encoded = enc.inference(inputs=[mel])[0]; enc_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter(); audio = adapter.inference(inputs=[encoded])[0]; adapter_ms = (time.perf_counter()-t)*1000
        if len(audio) != len(audio_positions):
            raise RuntimeError(f"task {task['index']}: {len(audio)} adapter tokens for {len(audio_positions)} placeholders")
        embeds = np.asarray(self.table[ids], np.float32)[None, :, :]
        embeds[0, audio_positions, :] = audio
        t = time.perf_counter(); prefill_out = prefill.inference(inputs=[embeds]); prefill_ms = (time.perf_counter()-t)*1000
        # Preserve only ordinary NumPy tensors before unmapping the three large
        # frontend graphs.  This is necessary before mapping decoder blocks.
        self.release_frontends()
        kvs, heads = self.decoder()
        decoder_rss = rss_mib()
        initial = int(prefill_out[1].shape[-2])
        capacity = next((value for value in self.capacities if initial < value), None)
        if capacity is None:
            raise RuntimeError(f"task {task['index']}: prefill {initial} exceeds capacities {self.capacities}")
        keys, values = [], []
        for layer in range(8):
            key = np.zeros((8, capacity, 64), np.float32); value = np.zeros_like(key)
            key[:, :initial] = prefill_out[1 + 2 * layer].reshape(8, initial, 64)
            value[:, :initial] = prefill_out[2 + 2 * layer].reshape(8, initial, 64)
            keys.append(key); values.append(value)
        inv_freq = np.load(input_dir / "rotary_inv_freq.npy").astype("float32")
        token = int(np.asarray(prefill_out[0]).argmax()); tokens = [token]
        decoder_npu_ms = cpu_cache_ms = 0.0; switches: list[dict[str, object]] = []
        for step in range(max_new):
            if step and token in eos:
                break
            position = initial + step
            if position >= capacity:
                next_capacity = next((value for value in self.capacities if value > capacity), None)
                if next_capacity is None:
                    raise RuntimeError(f"task {task['index']}: decoder reached unbuilt capacity above {capacity}")
                t = time.perf_counter()
                keys = [np.pad(value, ((0, 0), (0, next_capacity-capacity), (0, 0))) for value in keys]
                values = [np.pad(value, ((0, 0), (0, next_capacity-capacity), (0, 0))) for value in values]
                copy_ms = (time.perf_counter()-t)*1000
                switches.append({"at_position": position, "from": capacity, "to": next_capacity, "cpu_cache_copy_ms": copy_ms})
                capacity = next_capacity
            x = np.asarray(self.table[token:token+1], np.float32).reshape(1, 1, 512)
            angle = np.concatenate((inv_freq * position, inv_freq * position))[None, None, :]
            cos, sin = np.cos(angle).astype("float32"), np.sin(angle).astype("float32")
            mask = np.full((8, 1, capacity), -10000, np.float32); mask[:, :, :position+1] = 0
            for layer in range(8):
                t = time.perf_counter(); delta_k, delta_v = kvs[layer].inference(inputs=[x, cos, sin]); decoder_npu_ms += (time.perf_counter()-t)*1000
                t = time.perf_counter(); keys[layer][:, position:position+1] = delta_k; values[layer][:, position:position+1] = delta_v; cpu_cache_ms += (time.perf_counter()-t)*1000
                t = time.perf_counter(); x = self.block(capacity)[layer].inference(inputs=[x, keys[layer], values[layer], cos, sin, mask])[0]; decoder_npu_ms += (time.perf_counter()-t)*1000
            logits = []
            for head in heads:
                t = time.perf_counter(); logits.append(head.inference(inputs=[x])[0]); decoder_npu_ms += (time.perf_counter()-t)*1000
            token = int(np.concatenate(logits, axis=-1).argmax()); tokens.append(token)
        return {"index": task["index"], "profile": task["profile"], "tokens": tokens,
                "complete_to_eos": tokens[-1] in eos, "initial_prefill": initial, "final_bucket": capacity,
                "bucket_switches": switches, "latency_ms": {"encoder": enc_ms, "adapter": adapter_ms,
                "prefill": prefill_ms, "decoder_npu_total": decoder_npu_ms, "cpu_kv_total": cpu_cache_ms,
                "neural_total": enc_ms+adapter_ms+prefill_ms+decoder_npu_ms+cpu_cache_ms,
                "task_wall_total": (time.perf_counter()-started)*1000},
                "rss_mib": {"frontend_stage": frontend_rss, "decoder_stage": decoder_rss}}

    def close(self) -> None:
        self.release_frontends()
        self.release_decoder()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--token-embeddings", type=Path, required=True)
    parser.add_argument("--eos-id", type=int, default=151645)
    parser.add_argument("--max-new", type=int, default=400)
    parser.add_argument("--capacities", default="128,256,512")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capacities = tuple(int(value) for value in args.capacities.split(","))
    if tuple(sorted(set(capacities))) != capacities:
        raise ValueError("capacities must be strictly increasing")
    tasks = json.loads(args.inputs_manifest.read_text(encoding="utf-8"))
    worker = Worker(args.asset_root, args.token_embeddings, capacities)
    load_rss = rss_mib(); start = time.perf_counter()
    try:
        results = [worker.run(task, args.max_new, {args.eos_id}) for task in tasks]
    finally:
        worker.close()
    payload = {"version": 1, "worker": "persistent-rknn", "tasks": results,
               "summary": {"segments": len(results), "all_eos": all(row["complete_to_eos"] for row in results),
               "neural_total_ms": sum(row["latency_ms"]["neural_total"] for row in results),
               "worker_wall_ms": (time.perf_counter()-start)*1000, "rss_after_initial_load_mib": load_rss,
               "rss_before_release_mib": rss_mib()}}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
