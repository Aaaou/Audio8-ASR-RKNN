"""Generate deterministic original-model references for duration-controlled ASR WAVs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT = "Please transcribe this audio."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new", type=int, default=400)
    a = p.parse_args()

    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    conversation = [{"role": "user", "content": [
        {"type": "audio", "path": str(a.audio)},
        {"type": "text", "text": PROMPT},
    ]}]
    batch = dict(processor.apply_chat_template(
        conversation, return_tensors="pt", sampling_rate=16000,
        audio_padding="max_length", add_generation_prompt=True,
        audio_max_length=128000,
        text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
    ))
    prompt_tokens = int(batch["input_ids"].shape[1])
    t = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=a.max_new, do_sample=False)
    elapsed_ms = (time.perf_counter() - t) * 1000
    ids = generated[0, prompt_tokens:].tolist()
    payload = {
        "backend": "original PyTorch CPU FP32 eager",
        "audio": str(a.audio),
        "prefill_sequence": prompt_tokens,
        "tokens": ids,
        "complete_to_eos": bool(ids and ids[-1] == model.config.eos_token_id),
        "eos_id": model.config.eos_token_id,
        "text": processor.decode(ids, skip_special_tokens=True).strip(),
        "generation_wall_ms": elapsed_ms,
    }
    a.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
