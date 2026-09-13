"""Run FP16 RKNN repeatedly and expose a stable interval for external NPU-load sampling."""

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def cpu_seconds() -> float:
    data = Path(f"/proc/{os.getpid()}/stat").read_text().split()
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    return (int(data[13]) + int(data[14])) / ticks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = np.load(args.input).astype(np.float32, copy=False)
    rknn = RKNNLite(verbose=True)
    if rknn.load_rknn(str(args.rknn)) != 0 or rknn.init_runtime() != 0:
        raise RuntimeError("RKNN init failed")
    rknn.inference(inputs=[value])
    samples = []
    stop = threading.Event()

    def sample_npu() -> None:
        while not stop.is_set():
            try:
                load = Path("/sys/kernel/debug/rknpu/load").read_text().strip()
                freq = Path("/sys/kernel/debug/rknpu/freq").read_text().strip()
                values = [int(x) for x in re.findall(r"(\d+)%", load)]
                samples.append({"load": load, "freq_hz": int(freq), "max_load_percent": max(values, default=0)})
            except Exception as exc:
                samples.append({"error": str(exc)})
            time.sleep(0.02)

    sampler = threading.Thread(target=sample_npu, daemon=True)
    sampler.start()
    wall_start, cpu_start = time.perf_counter(), cpu_seconds()
    for _ in range(args.runs):
        out = rknn.inference(inputs=[value])[0]
        if not np.isfinite(out).all():
            raise RuntimeError("non-finite output")
    wall_s, cpu_s = time.perf_counter() - wall_start, cpu_seconds() - cpu_start
    stop.set()
    sampler.join(timeout=1)
    payload = {
        "pid": os.getpid(), "runs": args.runs, "wall_s": wall_s, "process_cpu_s": cpu_s,
        "process_cpu_utilization": cpu_s / wall_s * 100, "mean_inference_ms": wall_s / args.runs * 1000,
        "npu_load_samples": len(samples),
        "npu_peak_load_percent": max((sample.get("max_load_percent", 0) for sample in samples), default=0),
        "npu_active_samples": sum(sample.get("max_load_percent", 0) > 0 for sample in samples),
        "npu_peak_freq_hz": max((sample.get("freq_hz", 0) for sample in samples), default=0),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    rknn.release()


if __name__ == "__main__":
    main()
