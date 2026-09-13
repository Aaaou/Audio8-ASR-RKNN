"""Create per-segment Audio8 inputs under the deployment max-length contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT = "Please transcribe this audio."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.plan_dir / "plan.json").read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    # Loading the model supplies the canonical audio token id and RoPE values.
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                  torch_dtype=torch.float32,
                                                  attn_implementation="eager").eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for segment in plan["segments"]:
        index = int(segment["index"])
        bucket = float(segment["bucket_seconds"])
        wav = args.plan_dir / f"segment_{index:05d}_{segment['profile']}.wav"
        conversation = [{"role": "user", "content": [
            {"type": "audio", "path": str(wav)}, {"type": "text", "text": PROMPT},
        ]}]
        batch = dict(processor.apply_chat_template(
            conversation, return_tensors="pt", sampling_rate=16000,
            audio_padding="max_length", audio_max_length=round(bucket * 16000),
            add_generation_prompt=True,
            text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
        ))
        target = args.output_dir / f"segment_{index:05d}"
        target.mkdir(exist_ok=True)
        np.save(target / "input_features.npy", batch["input_features"][0].float().numpy())
        np.save(target / "input_ids.npy", batch["input_ids"].cpu().numpy().astype(np.int64))
        positions = (batch["input_ids"][0] == model.config.audio_token_id).nonzero().cpu().numpy().astype(np.int64)
        np.save(target / "audio_positions.npy", positions)
        np.save(target / "rotary_inv_freq.npy", model.language_model.model.rotary_emb.inv_freq.cpu().float().numpy())
        manifest.append({"index": index, "profile": segment["profile"], "input_dir": str(target),
                         "mel_shape": list(batch["input_features"][0].shape),
                         "prefill_sequence": int(batch["input_ids"].shape[1]),
                         "audio_placeholders": int(len(positions))})
    (args.output_dir / "inputs-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": len(manifest), "manifest": str(args.output_dir/'inputs-manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
