"""Evaluate Audio8 with both the encoder and audio adapter on RK3576 NPU.

The Qwen2 decoder intentionally remains the upstream PyTorch implementation.
Replacing modules instead of bypassing ``model.generate`` preserves ArkASR's
input-id bookkeeping, audio injection, prefill and KV-cache decode path.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput

PROMPT = "Please transcribe this audio."


def normalize_english(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


class _RKNNBase(torch.nn.Module):
    def __init__(self, rknn, config=None) -> None:
        super().__init__()
        self.rknn, self.config = rknn, config
        self.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.last_inference_s = 0.0

    def infer(self, value: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        output = np.asarray(self.rknn.inference(inputs=[value.detach().float().cpu().numpy()])[0], dtype=np.float32)
        self.last_inference_s += time.perf_counter() - start
        return torch.from_numpy(output).to(device=value.device, dtype=self.anchor.dtype)


class RKNNEncoderAdapter(_RKNNBase):
    def forward(self, input_features, feature_lens=None):
        mel = input_features[0] if input_features.ndim == 3 and input_features.shape[0] == 1 else input_features
        if tuple(mel.shape) != (128, 800):
            raise RuntimeError(f"expected [128,800] mel, got {tuple(mel.shape)}")
        return BaseModelOutput(last_hidden_state=self.infer(mel))


class RKNNAudioAdapter(_RKNNBase):
    """Replaces the MLP tower; it also performs the fixed 104->100 pooling."""
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if tuple(hidden_states.shape) != (104, 1024):
            raise RuntimeError(f"expected [104,1024] encoder hidden states, got {tuple(hidden_states.shape)}")
        return self.infer(hidden_states)


class ParameterIdentity(torch.nn.Module):
    """Identity with a parameter so the upstream dtype/device lookup remains valid."""
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder-rknn", type=Path, required=True)
    parser.add_argument("--adapter-rknn", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-audio-seconds", type=int, default=8)
    args = parser.parse_args()
    from rknnlite.api import RKNNLite
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
        torch_dtype=torch.float32, attn_implementation="eager").eval()
    encoder_rknn, adapter_rknn = RKNNLite(verbose=False), RKNNLite(verbose=False)
    try:
        for engine, path in ((encoder_rknn, args.encoder_rknn), (adapter_rknn, args.adapter_rknn)):
            if engine.load_rknn(str(path)) != 0 or engine.init_runtime() != 0:
                raise RuntimeError(f"Cannot initialize {path}")
        original_encoder = model.audio_encoder
        model.audio_encoder = RKNNEncoderAdapter(encoder_rknn, original_encoder.config)
        model.audio_mlp_tower = RKNNAudioAdapter(adapter_rknn)
        model.audio_projector = ParameterIdentity()
        rows = json.loads(args.manifest.read_text(encoding="utf-8"))[:args.limit]
        results, totals = [], [0, 0, 0, 0]
        for row in rows:
            conversation = [{"role": "user", "content": [{"type": "audio", "path": row["audio"]},
                            {"type": "text", "text": PROMPT}]}]
            batch = dict(processor.apply_chat_template(conversation, return_tensors="pt", sampling_rate=16000,
                audio_padding="max_length", add_generation_prompt=True, audio_max_length=args.max_audio_seconds * 16000,
                text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000}))
            model.audio_encoder.last_inference_s = model.audio_mlp_tower.last_inference_s = 0.0
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(**batch, max_new_tokens=128, do_sample=False)
            total_s = time.perf_counter() - start
            encoder_s, adapter_s = model.audio_encoder.last_inference_s, model.audio_mlp_tower.last_inference_s
            text = processor.decode(output_ids[0, int(batch["input_ids"].shape[1]):], skip_special_tokens=True).strip()
            reference, hypothesis = normalize_english(row["reference"]), normalize_english(text)
            words, chars = Levenshtein.distance(reference.split(), hypothesis.split()), Levenshtein.distance(reference.replace(" ", ""), hypothesis.replace(" ", ""))
            totals[0] += words; totals[1] += len(reference.split()); totals[2] += chars; totals[3] += len(reference.replace(" ", ""))
            results.append({**row, "hypothesis": text, "encoder_s": encoder_s, "adapter_s": adapter_s,
                            "qwen2_decoder_s": total_s-encoder_s-adapter_s, "generated_tokens": int(output_ids.shape[1]-batch["input_ids"].shape[1]),
                            "wer": words/max(len(reference.split()), 1), "cer": chars/max(len(reference.replace(" ", "")), 1)})
        payload = {"backend": "RK3576 FP16 RKNN encoder + FP16 RKNN audio MLP tower/projector + PyTorch CPU FP32 Qwen2 decoder",
                   "fixed_audio_seconds": args.max_audio_seconds, "samples": results,
                   "aggregate": {"wer": totals[0]/max(totals[1], 1), "cer": totals[2]/max(totals[3], 1),
                   "mean_encoder_s": float(np.mean([x["encoder_s"] for x in results])), "mean_adapter_s": float(np.mean([x["adapter_s"] for x in results])),
                   "mean_qwen2_decoder_s": float(np.mean([x["qwen2_decoder_s"] for x in results]))}}
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        encoder_rknn.release(); adapter_rknn.release()


if __name__ == "__main__":
    main()
