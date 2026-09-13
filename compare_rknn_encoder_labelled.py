"""Compare FP16 RKNN encoder output to the original CPU FP32 encoder on labelled inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rknnlite.api import RKNNLite
from transformers import AutoModelForCausalLM, AutoProcessor


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float32, copy=False)
    actual = actual.astype(np.float32, copy=False)
    return {
        "cosine": float(np.dot(reference.ravel(), actual.ravel()) / max(
            float(np.linalg.norm(reference)) * float(np.linalg.norm(actual)), 1e-12
        )),
        "normalized_l2": float(np.linalg.norm(reference - actual) / max(float(np.linalg.norm(reference)), 1e-12)),
        "mae": float(np.mean(np.abs(reference - actual))),
        "max_abs": float(np.max(np.abs(reference - actual))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(args.rknn)) != 0 or rknn.init_runtime() != 0:
        raise RuntimeError("RKNN initialization failed")
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = []
    try:
        for row in rows:
            conversation = [{"role": "user", "content": [
                {"type": "audio", "path": row["audio"]},
                {"type": "text", "text": "Please transcribe this audio."},
            ]}]
            batch = processor.apply_chat_template(
                conversation, return_tensors="pt", sampling_rate=16000,
                audio_padding="max_length", add_generation_prompt=True,
                audio_max_length=8 * 16000,
                text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
            )
            mel = batch["input_features"][0].detach().float().cpu().numpy()
            with torch.inference_mode():
                reference = model.audio_encoder(
                    torch.from_numpy(mel), feature_lens=torch.tensor([800], dtype=torch.long)
                ).last_hidden_state.detach().float().cpu().numpy()
            actual = np.asarray(rknn.inference(inputs=[mel])[0], dtype=np.float32)
            results.append({"index": row["index"], "id": row["id"], **metrics(reference, actual)})
    finally:
        rknn.release()
    payload = {"reference": "original Audio8-ASR CPU FP32 audio encoder", "candidate": "RK3576 FP16 RKNN encoder", "samples": results}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
