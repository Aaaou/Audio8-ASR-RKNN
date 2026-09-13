"""Export the fixed Audio8 audio MLP-tower/projector graph to ONNX.

This deliberately starts the decoder migration at its stateless boundary.  The
input is the 8-second encoder's [104, 1024] output and the graph performs the
four residual MLP blocks, the exact adaptive pooling used by ArkASR, then the
projector.  Its [100, 512] output replaces the audio-placeholder embeddings in
the original generation path; Qwen2 and its KV cache remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from transformers import AutoModelForCausalLM


class StaticAudioAdapterBucket(torch.nn.Module):
    def __init__(self, tower: torch.nn.Module, projector: torch.nn.Module, tokens: int) -> None:
        super().__init__()
        self.tower = tower
        self.projector = projector
        self.tokens = int(tokens)
        # ONNX's PyTorch exporter does not support adaptive_avg_pool1d when
        # 104 is not divisible by 100.  For this intentionally static bucket,
        # construct its exact bin-average matrix once.  Row j is the interval
        # [floor(j*104/100), ceil((j+1)*104/100)), exactly matching PyTorch.
        pooling = torch.zeros(self.tokens, 104, dtype=torch.float32)
        for token in range(self.tokens):
            begin = (token * 104) // self.tokens
            end = ((token + 1) * 104 + self.tokens - 1) // self.tokens
            pooling[token, begin:end] = 1.0 / (end - begin)
        self.register_buffer("pooling", pooling, persistent=False)

    def forward(self, audio_hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.tower(audio_hidden)
        hidden = self.pooling.to(dtype=hidden.dtype) @ hidden
        return self.projector(hidden)


def save_npy(path: Path, value: torch.Tensor | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    np.save(path, np.ascontiguousarray(value, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder-references", type=Path, required=True,
                        help="directory containing sample_*_reference.npy from the 8 s encoder export")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--calibration-samples", type=int, default=8)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    adapter = StaticAudioAdapterBucket(model.audio_mlp_tower, model.audio_projector, args.tokens).eval()
    source_files = sorted(args.encoder_references.glob("sample_*_reference.npy"))[: args.calibration_samples]
    if not source_files:
        raise FileNotFoundError(f"no encoder reference tensors in {args.encoder_references}")

    output = args.output
    calibration_dir = output / "calibration"
    samples: list[dict[str, object]] = []
    dataset_lines: list[str] = []
    for index, source_path in enumerate(source_files):
        source = np.load(source_path).astype(np.float32, copy=False)
        if source.shape != (104, 1024):
            raise ValueError(f"{source_path}: expected [104,1024], got {source.shape}")
        with torch.inference_mode():
            reference = adapter(torch.from_numpy(source))
        input_path = calibration_dir / f"sample_{index:02d}_input.npy"
        reference_path = calibration_dir / f"sample_{index:02d}_reference.npy"
        save_npy(input_path, source)
        save_npy(reference_path, reference)
        dataset_lines.append(str(input_path))
        samples.append({"index": index, "source": str(source_path), "input": str(input_path),
                        "reference": str(reference_path), "input_shape": list(source.shape),
                        "output_shape": list(reference.shape)})

    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / f"audio_adapter_h104_t{args.tokens}.onnx"
    example = torch.from_numpy(np.load(samples[0]["input"]))
    torch.onnx.export(adapter, (example,), onnx_path, input_names=["audio_hidden"],
                      output_names=["audio_embeddings"], opset_version=19,
                      do_constant_folding=True, dynamo=False)
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    (output / "dataset.txt").write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "model": str(args.model), "onnx": str(onnx_path), "tokens": args.tokens,
        "dataset": str(output / "dataset.txt"), "samples": samples,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"onnx": str(onnx_path), "samples": len(samples), "output_shape": samples[0]["output_shape"]}))


if __name__ == "__main__":
    main()
