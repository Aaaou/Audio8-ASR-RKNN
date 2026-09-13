"""Export one fixed-length Audio8-ASR audio-encoder bucket and real calibration data.

The upstream encoder contains Python-driven chunking and variable-length packing.
This tool intentionally traces one fixed mel-frame bucket so the RKNN graph has
no dynamic audio-length contract. The CPU wrapper remains responsible for
resampling, segmentation, padding, and choosing a bucket at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import onnx
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


class StaticAudioEncoderBucket(torch.nn.Module):
    """Equivalent fixed-bucket form of the upstream encoder's dynamic packing."""

    def __init__(self, audio_encoder: torch.nn.Module, frames: int) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.frames = frames
        self.chunks = frames // 100
        self.frames_per_chunk = 100
        self.tokens_per_chunk = 13
        self.total_tokens = self.chunks * self.tokens_per_chunk
        self.register_buffer(
            "cu_seqlens",
            torch.tensor([0, self.total_tokens], dtype=torch.int32),
            persistent=False,
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # The upstream fixed 100-frame chunks all reduce to 13 frames after
        # its three stride-2 Conv2d layers. This replaces pad_sequence and
        # boolean packing with its exact static equivalent.
        chunks = input_features.transpose(0, 1).reshape(
            self.chunks, self.frames_per_chunk, 128
        )
        hidden = chunks.transpose(1, 2).unsqueeze(1)
        encoder = self.audio_encoder
        hidden = torch.nn.functional.gelu(encoder.conv2d1(hidden))
        hidden = torch.nn.functional.gelu(encoder.conv2d2(hidden))
        hidden = torch.nn.functional.gelu(encoder.conv2d3(hidden))
        batch, channels, freq, time = hidden.shape
        hidden = encoder.conv_out(
            hidden.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * freq)
        )
        hidden = hidden + encoder.positional_embedding.positional_embedding[:time, :].unsqueeze(0).to(hidden.dtype)
        hidden = hidden.reshape(self.total_tokens, hidden.shape[-1])
        for layer in encoder.layers:
            hidden = layer(hidden, self.cu_seqlens)[0]
        hidden = encoder.ln_post(hidden)
        hidden = encoder.proj1(hidden)
        hidden = encoder.act(hidden)
        return encoder.proj2(hidden)


def save_npy(path: Path, value: torch.Tensor | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    np.save(path, np.ascontiguousarray(value, dtype=np.float32))


def make_feature(processor, audio: np.ndarray, frames: int) -> torch.Tensor:
    extracted = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=frames * 160,
    )
    value = extracted.input_features[0]
    if tuple(value.shape) != (128, frames):
        raise RuntimeError(f"expected mel [128,{frames}], got {tuple(value.shape)}")
    return value.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument("--calibration-samples", type=int, default=8)
    args = parser.parse_args()

    if args.frames <= 0 or args.frames % 100:
        raise ValueError("--frames must be a positive multiple of 100 for this fixed chunk bucket")

    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    encoder = StaticAudioEncoderBucket(model.audio_encoder, args.frames).eval()

    wav_parts = [librosa.load(audio_path, sr=16000, mono=True)[0] for audio_path in args.audio]
    wav = np.concatenate(wav_parts).astype(np.float32, copy=False)
    samples_per_bucket = args.frames * 160
    if wav.size < samples_per_bucket:
        wav = np.pad(wav, (0, samples_per_bucket - wav.size))

    output = args.output
    calibration_dir = output / "calibration"
    dataset_lines: list[str] = []
    refs: list[dict[str, object]] = []
    for index in range(args.calibration_samples):
        max_start = max(0, wav.size - samples_per_bucket)
        start = 0 if args.calibration_samples == 1 else round(max_start * index / (args.calibration_samples - 1))
        clip = wav[start : start + samples_per_bucket]
        if clip.size < samples_per_bucket:
            clip = np.pad(clip, (0, samples_per_bucket - clip.size))
        features = make_feature(processor, clip, args.frames)
        with torch.inference_mode():
            reference = encoder(features)
        input_path = calibration_dir / f"sample_{index:02d}_input.npy"
        output_path = calibration_dir / f"sample_{index:02d}_reference.npy"
        save_npy(input_path, features)
        save_npy(output_path, reference)
        dataset_lines.append(str(input_path))
        refs.append(
            {
                "index": index,
                "start_samples": int(start),
                "input": str(input_path),
                "reference": str(output_path),
                "input_shape": list(features.shape),
                "output_shape": list(reference.shape),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / f"audio_encoder_f{args.frames}.onnx"
    example = torch.from_numpy(np.load(calibration_dir / "sample_00_input.npy"))
    with torch.inference_mode():
        upstream = model.audio_encoder(example).last_hidden_state
        static = encoder(example)
    max_error = float((upstream - static).abs().max())
    if max_error > 1e-5:
        raise RuntimeError(f"static bucket differs from upstream encoder: max_abs={max_error}")
    torch.onnx.export(
        encoder,
        (example,),
        onnx_path,
        input_names=["input_features"],
        output_names=["audio_hidden"],
        opset_version=19,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    (output / "dataset.txt").write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "model": str(args.model),
                "audio": [str(audio_path) for audio_path in args.audio],
                "frames": args.frames,
                "onnx": str(onnx_path),
                "dataset": str(output / "dataset.txt"),
                "static_vs_upstream_max_abs": max_error,
                "samples": refs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"onnx": str(onnx_path), "samples": len(refs), "output_shape": refs[0]["output_shape"]}))


if __name__ == "__main__":
    main()
