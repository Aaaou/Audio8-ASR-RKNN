"""Board-side correctness, latency and layer-placement probe for Audio8 adapter RKNN."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    actual64, reference64 = actual.astype(np.float64), reference.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "normalized_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference64)),
        "cosine": float(np.sum(actual64 * reference64) / (np.linalg.norm(actual64) * np.linalg.norm(reference64))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = sorted(args.calibration.glob("sample_*_input.npy"))
    if not inputs:
        raise FileNotFoundError("no calibration inputs")
    rknn = RKNNLite(verbose=True)
    try:
        if rknn.load_rknn(str(args.rknn)) != 0 or rknn.init_runtime() != 0:
            raise RuntimeError("RKNN initialization failed")
        # Warm-up is intentionally excluded.
        warmup = rknn.inference(inputs=[np.load(inputs[0]).astype(np.float32)])[0]
        rows = []
        elapsed = []
        for input_path in inputs:
            reference_path = input_path.with_name(input_path.name.replace("_input.npy", "_reference.npy"))
            source = np.load(input_path).astype(np.float32)
            reference = np.load(reference_path).astype(np.float32)
            for _ in range(args.runs):
                start = time.perf_counter()
                actual = np.asarray(rknn.inference(inputs=[source])[0], dtype=np.float32)
                elapsed.append(time.perf_counter() - start)
            rows.append({"input": str(input_path), "reference": str(reference_path),
                         "output_shape": list(actual.shape), **metrics(actual, reference)})
        payload = {"runs_per_sample": args.runs, "warmup_shape": list(np.asarray(warmup).shape),
                   "samples": rows, "latency_ms": {"mean": float(np.mean(elapsed) * 1e3),
                   "p50": float(np.percentile(elapsed, 50) * 1e3), "p95": float(np.percentile(elapsed, 95) * 1e3),
                   "min": float(np.min(elapsed) * 1e3), "max": float(np.max(elapsed) * 1e3)}}
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
