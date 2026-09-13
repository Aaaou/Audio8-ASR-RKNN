"""Export a fixed-context Qwen2 prefill logits-only RKNN compatibility gate.

It disables KV cache and emits only final-position logits. This is not yet a
deployable autoregressive decoder; it checks whether Qwen2's attention, MLP,
normalization and LM head can all compile before cache graphs are attempted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT = "Please transcribe this audio."


class Qwen2PrefillLastLogits(torch.nn.Module):
    def __init__(self, language_model: torch.nn.Module) -> None:
        super().__init__()
        self.language_model = language_model

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        output = self.language_model(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
        return output.logits[:, -1, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-audio-seconds", type=int, default=8)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
        torch_dtype=torch.float32, attn_implementation="eager").eval()
    conversation = [{"role": "user", "content": [{"type": "audio", "path": str(args.audio)},
                     {"type": "text", "text": PROMPT}]}]
    batch = dict(processor.apply_chat_template(conversation, return_tensors="pt", sampling_rate=16000,
        audio_padding="max_length", add_generation_prompt=True, audio_max_length=args.max_audio_seconds * 16000,
        text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000}))
    with torch.inference_mode():
        embeddings = model._inject_audio_embeddings(batch["input_ids"], batch["input_features"], batch.get("feature_lens"))
    wrapper = Qwen2PrefillLastLogits(model.language_model).eval()
    with torch.inference_mode():
        reference = wrapper(embeddings)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    input_path, ref_path = output / "input_embeddings.npy", output / "reference_last_logits.npy"
    np.save(input_path, embeddings.detach().cpu().float().numpy())
    np.save(ref_path, reference.detach().cpu().float().numpy())
    onnx_path = output / f"qwen2_prefill_s{embeddings.shape[1]}_last_logits.onnx"
    torch.onnx.export(wrapper, (embeddings,), onnx_path, input_names=["inputs_embeds"], output_names=["last_logits"],
        opset_version=19, do_constant_folding=True, dynamo=False)
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    manifest = {"onnx": str(onnx_path), "input": str(input_path), "reference": str(ref_path),
        "input_shape": list(embeddings.shape), "output_shape": list(reference.shape),
        "argmax": int(reference.argmax(dim=-1).item()), "input_ids_shape": list(batch["input_ids"].shape)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
