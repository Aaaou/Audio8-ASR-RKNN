"""Compare a deployed RKNN Audio8-ASR encoder bucket with CPU references."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | bool]:
    ref = reference.astype(np.float64).reshape(-1)
    out = actual.astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(ref))
    return {
        "finite": bool(np.isfinite(actual).all()),
        "cosine": float(np.dot(ref, out) / max(np.linalg.norm(ref) * np.linalg.norm(out), 1e-12)),
        "normalized_l2": float(np.linalg.norm(ref - out) / max(denom, 1e-12)),
        "mae": float(np.mean(np.abs(ref - out))),
        "max_abs": float(np.max(np.abs(ref - out))),
    }


def rss_mib() -> float:
    # Linux ru_maxrss is KiB. It is process-visible RSS only, not CMA/NPU memory.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-format", choices=("nchw", "none"), default="nchw")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rknn = RKNNLite(verbose=False)
    try:
        started = time.perf_counter()
        if rknn.load_rknn(str(args.model)) != 0:
            raise RuntimeError("load_rknn failed")
        load_s = time.perf_counter() - started
        rss_mib_after_load = rss_mib()
        started = time.perf_counter()
        if rknn.init_runtime() != 0:
            raise RuntimeError("init_runtime failed")
        init_s = time.perf_counter() - started
        rss_mib_after_init = rss_mib()
        rows = []
        for item in manifest["samples"]:
            source = np.load(item["input"]).astype(np.float32)
            reference = np.load(item["reference"]).astype(np.float32)
            inference_kwargs = {"inputs": [source]}
            if args.data_format == "nchw":
                inference_kwargs["data_format"] = ["nchw"]
            rknn.inference(**inference_kwargs)
            timings = []
            output = None
            for _ in range(args.repeat):
                started = time.perf_counter()
                output = np.asarray(rknn.inference(**inference_kwargs)[0])
                timings.append(time.perf_counter() - started)
            if output is None or output.shape != reference.shape:
                raise RuntimeError(f"unexpected output shape {None if output is None else output.shape}, expected {reference.shape}")
            row = {"index": item["index"], "mean_inference_s": float(np.mean(timings)), "min_inference_s": float(np.min(timings))}
            row.update(metrics(reference, output))
            rows.append(row)
        report = {
            "schema": "audio8-asr-rk3576-audio-encoder-v1",
            "model": str(args.model),
            "manifest": str(args.manifest),
            "load_s": load_s,
            "init_s": init_s,
            "rss_mib_after_load": rss_mib_after_load,
            "rss_mib_after_init": rss_mib_after_init,
            "rss_mib_peak": rss_mib(),
            "repeat": args.repeat,
            "data_format": args.data_format,
            "samples": rows,
        }
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
