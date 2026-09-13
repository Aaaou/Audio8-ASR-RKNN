"""Evaluate a fixed 8-second RKNN encoder with the original CPU decoder."""

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


class RKNNEncoderAdapter(torch.nn.Module):
    """Expose RKNN output through the custom model's expected audio-encoder API."""

    def __init__(self, rknn, config) -> None:
        super().__init__()
        self.rknn = rknn
        self.config = config
        self.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.last_inference_s = 0.0

    def forward(self, input_features, feature_lens=None):
        mel = input_features.detach().float().cpu().numpy()
        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel[0]
        if mel.shape != (128, 800):
            raise RuntimeError(f"expected fixed [128,800] mel, got {mel.shape}")
        start = time.perf_counter()
        output = np.asarray(self.rknn.inference(inputs=[mel])[0], dtype=np.float32)
        self.last_inference_s += time.perf_counter() - start
        hidden = torch.from_numpy(output).to(device=input_features.device, dtype=self.anchor.dtype)
        return BaseModelOutput(last_hidden_state=hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rknn", type=Path)
    parser.add_argument("--encoder", choices=("rknn", "cpu"), default="rknn")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-audio-seconds", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cpu")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()
    rknn = None
    if args.encoder == "rknn":
        from rknnlite.api import RKNNLite

        if args.rknn is None:
            raise ValueError("--rknn is required when --encoder=rknn")
        rknn = RKNNLite(verbose=False)
        if rknn.load_rknn(str(args.rknn)) != 0 or rknn.init_runtime() != 0:
            raise RuntimeError("Unable to initialize the RKNN encoder")
        original_encoder = model.audio_encoder
        model.audio_encoder = RKNNEncoderAdapter(rknn, original_encoder.config)
        del original_encoder

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))[: args.limit]
    results = []
    total_word_errors = total_ref_words = total_char_errors = total_ref_chars = 0
    try:
        for row in rows:
            conversation = [{"role": "user", "content": [
                {"type": "audio", "path": row["audio"]},
                {"type": "text", "text": PROMPT},
            ]}]
            batch = processor.apply_chat_template(
                conversation,
                return_tensors="pt",
                sampling_rate=16000,
                audio_padding="max_length",
                add_generation_prompt=True,
                audio_max_length=args.max_audio_seconds * 16000,
                text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
            )
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(batch).items()}
            if args.encoder != "rknn":
                raise ValueError("This runner's CPU-injection control is performed separately.")
            model.audio_encoder.last_inference_s = 0.0
            generate_start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(**batch, max_new_tokens=128, do_sample=False)
            total_s = time.perf_counter() - generate_start
            encoder_s = model.audio_encoder.last_inference_s
            decoder_s = total_s - encoder_s
            prompt_len = int(batch["input_ids"].shape[1])
            text = processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
            reference = normalize_english(row["reference"])
            hypothesis = normalize_english(text)
            ref_words, hyp_words = reference.split(), hypothesis.split()
            ref_chars, hyp_chars = reference.replace(" ", ""), hypothesis.replace(" ", "")
            word_errors = Levenshtein.distance(ref_words, hyp_words)
            char_errors = Levenshtein.distance(ref_chars, hyp_chars)
            total_word_errors += word_errors
            total_ref_words += len(ref_words)
            total_char_errors += char_errors
            total_ref_chars += len(ref_chars)
            results.append({
                **row, "hypothesis": text, "encoder_s": encoder_s, "decoder_s": decoder_s,
                "wer": word_errors / max(len(ref_words), 1),
                "cer": char_errors / max(len(ref_chars), 1),
            })
    finally:
        if rknn is not None:
            rknn.release()
    payload = {
        "backend": f"{args.encoder} audio encoder plus PyTorch CPU FP32 adapter/projector/Qwen2 decoder",
        "fixed_audio_seconds": args.max_audio_seconds,
        "samples": results,
        "aggregate": {
            "wer": total_word_errors / max(total_ref_words, 1),
            "cer": total_char_errors / max(total_ref_chars, 1),
            "mean_encoder_s": float(np.mean([row["encoder_s"] for row in results])),
            "mean_decoder_s": float(np.mean([row["decoder_s"] for row in results])),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
