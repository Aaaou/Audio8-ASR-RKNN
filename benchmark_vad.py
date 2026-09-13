"""Benchmark the energy baseline and Silero ONNX VAD on Audio8 inputs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from long_audio_segmenter import SAMPLE_RATE, speech_intervals


def stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {"mean": float(data.mean()), "p50": float(np.percentile(data, 50)), "p95": float(np.percentile(data, 95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--energy-threshold-db", type=float, default=-42.0)
    parser.add_argument("--silero-threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--merge-silence-ms", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    t = time.perf_counter(); silero = load_silero_vad(onnx=True); silero_load_ms = (time.perf_counter()-t)*1000
    records = []
    for path in args.audio:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{path}: expected 16 kHz, got {sr}")
        if audio.ndim == 2: audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio)
        energy_ms, silero_ms = [], []
        energy_segments = silero_segments = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            energy_segments = speech_intervals(audio, args.energy_threshold_db, 30,
                                                args.min_speech_ms, args.merge_silence_ms)
            energy_ms.append((time.perf_counter()-t)*1000)
            t = time.perf_counter()
            items = get_speech_timestamps(torch.from_numpy(audio), silero, sampling_rate=SAMPLE_RATE,
                                          threshold=args.silero_threshold,
                                          min_speech_duration_ms=args.min_speech_ms,
                                          min_silence_duration_ms=args.merge_silence_ms,
                                          return_seconds=False)
            silero_segments = [(int(x["start"]), int(x["end"])) for x in items]
            silero_ms.append((time.perf_counter()-t)*1000)
        def describe(items: list[tuple[int, int]]) -> dict[str, object]:
            return {"segments": len(items), "speech_seconds": sum(b-a for a,b in items)/SAMPLE_RATE,
                    "ranges_seconds": [[round(a/SAMPLE_RATE, 3), round(b/SAMPLE_RATE, 3)] for a,b in items]}
        records.append({"audio": str(path), "duration_seconds": len(audio)/SAMPLE_RATE,
                        "energy": {"latency_ms": stats(energy_ms), **describe(energy_segments)},
                        "silero_onnx_cpu": {"latency_ms": stats(silero_ms), **describe(silero_segments)}})
    payload = {"silero": {"version": "6.2.1", "runtime": "ONNX Runtime CPU", "cold_model_load_ms": silero_load_ms,
                            "threshold": args.silero_threshold}, "repeats": args.repeats, "records": records}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
