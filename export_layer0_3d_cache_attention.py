"""Export layer-0 decoder graphs with a rank-3, CPU-owned KV-cache contract.

Unlike the previous decode graph, this graph never receives cache tensors with
a batch dimension and never updates cache internally.  The host appends the
RoPE-applied K/V delta, then supplies [heads, sequence, head_dim] to the NPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT = "Please transcribe this audio."
HEADS, HEAD_DIM, HIDDEN = 8, 64, 512


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


class KVProjection3D(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.norm = layer.input_layernorm
        self.k_proj = layer.self_attn.k_proj
        self.v_proj = layer.self_attn.v_proj

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = self.norm(hidden)
        k = self.k_proj(x).reshape(HEADS, 1, HEAD_DIM)
        v = self.v_proj(x).reshape(HEADS, 1, HEAD_DIM)
        c, s = cos.reshape(1, 1, HEAD_DIM), sin.reshape(1, 1, HEAD_DIM)
        return k * c + rotate_half(k) * s, v


class ReadOnlyBlock3D(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.norm = layer.input_layernorm
        self.q_proj = layer.self_attn.q_proj
        self.o_proj = layer.self_attn.o_proj
        self.post_norm = layer.post_attention_layernorm
        self.gate = layer.mlp.gate_proj
        self.up = layer.mlp.up_proj
        self.down = layer.mlp.down_proj
        self.scale = layer.self_attn.scaling

    def forward(self, hidden: torch.Tensor, full_k: torch.Tensor, full_v: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        x = self.norm(hidden)
        q = self.q_proj(x).reshape(HEADS, 1, HEAD_DIM)
        c, s = cos.reshape(1, 1, HEAD_DIM), sin.reshape(1, 1, HEAD_DIM)
        q = q * c + rotate_half(q) * s
        score = torch.matmul(q, full_k.transpose(-2, -1)) * self.scale
        probability = torch.softmax(score.float(), dim=-1).to(q.dtype)
        attention = torch.matmul(probability, full_v).reshape(1, 1, HIDDEN)
        residual = hidden + self.o_proj(attention)
        mlp_in = self.post_norm(residual)
        return residual + self.down(F.silu(self.gate(mlp_in)) * self.up(mlp_in))


def save(path: Path, tensor: torch.Tensor) -> None:
    np.save(path, tensor.detach().cpu().float().numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    conversation = [{"role": "user", "content": [
        {"type": "audio", "path": str(args.audio)}, {"type": "text", "text": PROMPT}
    ]}]
    batch = dict(processor.apply_chat_template(
        conversation, return_tensors="pt", sampling_rate=16000, audio_padding="max_length",
        add_generation_prompt=True, audio_max_length=128000,
        text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
    ))
    with torch.inference_mode():
        embeds = model._inject_audio_embeddings(batch["input_ids"], batch["input_features"], batch.get("feature_lens"))
        prefill = model.language_model(inputs_embeds=embeds, use_cache=True, return_dict=True)
        token = prefill.logits[:, -1, :].argmax(-1, keepdim=True)
        hidden = model.get_input_embeddings()(token)
        sequence = int(prefill.past_key_values.layers[0].keys.shape[-2])
        position = torch.tensor([[sequence]], dtype=torch.long)
        cos, sin = model.language_model.model.rotary_emb(hidden, position)
        layer = model.language_model.model.layers[0]
        kv, block = KVProjection3D(layer).eval(), ReadOnlyBlock3D(layer).eval()
        key_delta, value_delta = kv(hidden, cos, sin)
        old = prefill.past_key_values.layers[0]
        full_k = torch.cat((old.keys.reshape(HEADS, sequence, HEAD_DIM), key_delta), dim=1).contiguous()
        full_v = torch.cat((old.values.reshape(HEADS, sequence, HEAD_DIM), value_delta), dim=1).contiguous()
        reference = block(hidden, full_k, full_v, cos, sin)
        # Compare exact manual semantics with the layer output from the original model.
        # This calls the native layer with the same *historical* cache.  It is
        # only a reference check; our exported graphs remain cache-stateless.
        original = layer(hidden, attention_mask=torch.zeros(1, 1, 1, sequence + 1),
                         position_ids=position, past_key_values=prefill.past_key_values,
                         cache_position=torch.tensor([sequence]), position_embeddings=(cos, sin),
                         use_cache=True)[0]
        relative = float(torch.linalg.vector_norm(reference - original) / torch.linalg.vector_norm(original))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in {"hidden": hidden, "cos": cos, "sin": sin, "full_k": full_k,
                        "full_v": full_v, "key_delta_reference": key_delta,
                        "value_delta_reference": value_delta, "hidden_reference": reference,
                        "original_layer_reference": original}.items():
        save(args.output / f"{name}.npy", value)
    torch.onnx.export(kv, (hidden, cos, sin), args.output / "kv_3d.onnx",
                      input_names=["hidden", "cos", "sin"], output_names=["key_delta", "value_delta"],
                      opset_version=19, dynamo=False)
    torch.onnx.export(block, (hidden, full_k, full_v, cos, sin), args.output / "block_3d.onnx",
                      input_names=["hidden", "full_k", "full_v", "cos", "sin"], output_names=["hidden_out"],
                      opset_version=19, dynamo=False)
    onnx.checker.check_model(onnx.load(args.output / "kv_3d.onnx"))
    onnx.checker.check_model(onnx.load(args.output / "block_3d.onnx"))
    (args.output / "manifest.json").write_text(json.dumps({"prefill_length": sequence, "token": int(token),
        "manual_vs_original_layer_l2": relative}, indent=2), encoding="utf-8")
    print((args.output / "manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
