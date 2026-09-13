"""Build an Audio8-ASR static ONNX bucket as RK3576 W8A8 or W4A16."""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantized-dtype", choices=("w8a8", "w4a16"), required=True)
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument("--no-quantization", action="store_true")
    args = parser.parse_args()

    rknn = RKNN(verbose=True)
    try:
        config = {
            "target_platform": "rk3576",
            "optimization_level": args.optimization_level,
            "quantized_dtype": args.quantized_dtype,
        }
        if rknn.config(**config) != 0:
            raise RuntimeError("rknn.config failed")
        if rknn.load_onnx(model=str(args.onnx.resolve())) != 0:
            raise RuntimeError("rknn.load_onnx failed")
        if rknn.build(
            do_quantization=not args.no_quantization,
            dataset=None if args.no_quantization else str(args.dataset.resolve()),
        ) != 0:
            raise RuntimeError("rknn.build failed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if rknn.export_rknn(str(args.output.resolve())) != 0:
            raise RuntimeError("rknn.export_rknn failed")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
