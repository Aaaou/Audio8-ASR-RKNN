"""Validate the stateless Audio8 adapter ONNX graph against PyTorch references."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.bucket / "manifest.json").read_text(encoding="utf-8"))
    graph = onnx.load(manifest["onnx"])
    session = ort.InferenceSession(manifest["onnx"], providers=["CPUExecutionProvider"])
    rows = []
    for item in manifest["samples"]:
        source = np.load(item["input"]).astype(np.float32)
        reference = np.load(item["reference"]).astype(np.float32)
        actual = session.run(None, {"audio_hidden": source})[0]
        rows.append({"index": item["index"], "max_abs": float(np.max(np.abs(actual-reference))),
                     "normalized_l2": float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
                     "cosine": float(np.sum(actual*reference)/(np.linalg.norm(actual)*np.linalg.norm(reference)))})
    print(json.dumps({"operators": dict(Counter(node.op_type for node in graph.graph.node)), "samples": rows}, indent=2))


if __name__ == "__main__":
    main()
