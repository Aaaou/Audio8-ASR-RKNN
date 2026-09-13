"""Benchmark the static Audio8 encoder with board-local ONNX Runtime CPU."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float32, copy=False)
    actual = actual.astype(np.float32, copy=False)
    denominator = max(float(np.linalg.norm(reference)), 1e-12)
    return {
        "cosine": float(np.dot(reference.ravel(), actual.ravel()) / max(
            float(np.linalg.norm(reference)) * float(np.linalg.norm(actual)), 1e-12
        )),
        "normalized_l2": float(np.linalg.norm(reference - actual) / denominator),
        "mae": float(np.mean(np.abs(reference - actual))),
        "max_abs": float(np.max(np.abs(reference - actual))),
    }


def rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.intra_op_threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rss_after_load = rss_mib()
    input_name = session.get_inputs()[0].name
    results = []

    for item in manifest["samples"]:
        source = np.load(item["input"]).astype(np.float16)
        reference = np.load(item["reference"]).astype(np.float32)
        session.run(None, {input_name: source})
        timings = []
        output = None
        for _ in range(args.runs):
            start = time.perf_counter()
            output = session.run(None, {input_name: source})[0]
            timings.append(time.perf_counter() - start)
        assert output is not None
        row = {
            "index": item["index"],
            "mean_inference_s": float(np.mean(timings)),
            "min_inference_s": float(np.min(timings)),
        }
        row.update(metrics(reference, output))
        results.append(row)

    payload = {
        "backend": "onnxruntime CPUExecutionProvider",
        "model": args.model.name,
        "input_type": session.get_inputs()[0].type,
        "output_type": session.get_outputs()[0].type,
        "runs_per_sample": args.runs,
        "intra_op_num_threads": args.intra_op_threads,
        "rss_mib_after_load": rss_after_load,
        "rss_mib_peak": rss_mib(),
        "samples": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
