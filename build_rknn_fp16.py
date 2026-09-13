"""Build a FP16 RKNN model without post-training quantization."""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimization-level", type=int, default=2)
    args = parser.parse_args()
    rknn = RKNN(verbose=True)
    try:
        if rknn.config(target_platform="rk3576", optimization_level=args.optimization_level) != 0:
            raise RuntimeError("rknn.config failed")
        if rknn.load_onnx(model=str(args.onnx.resolve())) != 0:
            raise RuntimeError("rknn.load_onnx failed")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError("rknn.build FP16 failed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if rknn.export_rknn(str(args.output.resolve())) != 0:
            raise RuntimeError("rknn.export_rknn failed")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
