"""Measure labelled end-to-end Audio8-ASR CPU transcription quality and latency."""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoModelForCausalLM, AutoProcessor


PROMPT = "Please transcribe this audio."


def normalize_english(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def transcript(processor, model, device: torch.device, audio_path: str, max_audio_seconds: int) -> tuple[str, float]:
    conversation = [{"role": "user", "content": [
        {"type": "audio", "path": audio_path},
        {"type": "text", "text": PROMPT},
    ]}]
    batch = processor.apply_chat_template(
        conversation,
        return_tensors="pt",
        sampling_rate=16000,
        audio_padding="longest",
        add_generation_prompt=True,
        audio_max_length=max_audio_seconds * 16000,
        text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
    )
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(batch).items()}
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**batch, max_new_tokens=128, do_sample=False)
    elapsed = time.perf_counter() - start
    prompt_len = int(batch["input_ids"].shape[1])
    text = processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
    return text, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-audio-seconds", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cpu")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))[: args.limit]
    results = []
    total_ref_words = total_word_errors = total_ref_chars = total_char_errors = 0
    for row in rows:
        hypothesis, elapsed_s = transcript(processor, model, device, row["audio"], args.max_audio_seconds)
        reference = normalize_english(row["reference"])
        hypothesis_normalized = normalize_english(hypothesis)
        ref_words, hyp_words = reference.split(), hypothesis_normalized.split()
        ref_chars = reference.replace(" ", "")
        hyp_chars = hypothesis_normalized.replace(" ", "")
        word_errors = Levenshtein.distance(ref_words, hyp_words)
        char_errors = Levenshtein.distance(ref_chars, hyp_chars)
        total_ref_words += len(ref_words)
        total_word_errors += word_errors
        total_ref_chars += len(ref_chars)
        total_char_errors += char_errors
        results.append({
            **row,
            "hypothesis": hypothesis,
            "reference_normalized": reference,
            "hypothesis_normalized": hypothesis_normalized,
            "wer": word_errors / max(len(ref_words), 1),
            "cer": char_errors / max(len(ref_chars), 1),
            "elapsed_s": elapsed_s,
            "rtf": elapsed_s / row["duration_s"],
        })
    payload = {
        "backend": "PyTorch CPU FP32 original model",
        "normalization": "lowercase, punctuation removed, whitespace collapsed; CER excludes spaces",
        "samples": results,
        "aggregate": {
            "wer": total_word_errors / max(total_ref_words, 1),
            "cer": total_char_errors / max(total_ref_chars, 1),
            "total_audio_s": sum(row["duration_s"] for row in results),
            "total_transcribe_s": sum(row["elapsed_s"] for row in results),
            "rtf": sum(row["elapsed_s"] for row in results) / max(sum(row["duration_s"] for row in results), 1e-12),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
