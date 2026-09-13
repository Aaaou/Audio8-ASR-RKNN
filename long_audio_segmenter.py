"""Plan bounded Audio8-ASR work items from a 16 kHz mono WAV file.

This is deliberately an application-layer component.  Audio8 is not trained
as a cross-chunk streaming encoder, so every emitted work item is an
independent ASR request with a fresh prefill and KV cache.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000
PROFILES = (
    (8.0, "s8", 8.0),
    (16.0, "s16", 16.0),
    (30.0, "s30", 30.0),
)


@dataclass(frozen=True)
class Segment:
    index: int
    start_sample: int
    end_sample: int
    profile: str
    bucket_seconds: float
    forced_boundary: bool

    def json(self) -> dict[str, object]:
        result = asdict(self)
        result["start_ms"] = round(self.start_sample * 1000 / SAMPLE_RATE, 3)
        result["end_ms"] = round(self.end_sample * 1000 / SAMPLE_RATE, 3)
        result["duration_ms"] = round((self.end_sample - self.start_sample) * 1000 / SAMPLE_RATE, 3)
        return result


def choose_profile(seconds: float) -> tuple[str, float]:
    for limit, name, bucket in PROFILES:
        if seconds <= limit:
            return name, bucket
    raise ValueError(f"segment {seconds:.3f}s exceeds 30-second maximum")


def speech_intervals(samples: np.ndarray, threshold_db: float, frame_ms: int, min_speech_ms: int,
                     merge_silence_ms: int) -> list[tuple[int, int]]:
    """A deterministic energy VAD fallback; replaceable by a neural VAD later."""
    frame = SAMPLE_RATE * frame_ms // 1000
    if frame <= 0:
        raise ValueError("frame_ms must produce a non-empty frame")
    padded = np.pad(samples, (0, (-len(samples)) % frame))
    rms = np.sqrt(np.mean(padded.reshape(-1, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    active = db >= threshold_db
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        if not is_active and start is not None:
            runs.append((start * frame, min(index * frame, len(samples))))
            start = None
    if start is not None:
        runs.append((start * frame, len(samples)))

    minimum = SAMPLE_RATE * min_speech_ms // 1000
    gap = SAMPLE_RATE * merge_silence_ms // 1000
    kept = [(a, b) for a, b in runs if b - a >= minimum]
    merged: list[tuple[int, int]] = []
    for a, b in kept:
        if merged and a - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def split_interval(start: int, end: int, preferred_seconds: float, maximum_seconds: float,
                   overlap_seconds: float) -> list[tuple[int, int, bool]]:
    preferred = round(preferred_seconds * SAMPLE_RATE)
    maximum = round(maximum_seconds * SAMPLE_RATE)
    overlap = round(overlap_seconds * SAMPLE_RATE)
    if end - start <= maximum:
        return [(start, end, False)]
    # Fixed boundaries are intentional here: VAD already chose natural outer
    # boundaries.  The overlap protects words that straddle an inner boundary.
    result: list[tuple[int, int, bool]] = []
    cursor = start
    while cursor < end:
        right = min(cursor + preferred, end)
        result.append((cursor, right, right < end))
        if right == end:
            break
        cursor = right - overlap
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vad-threshold-db", type=float, default=-42.0)
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--merge-silence-ms", type=int, default=300)
    parser.add_argument("--preferred-seconds", type=float, default=16.0)
    parser.add_argument("--maximum-seconds", type=float, default=24.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.6)
    args = parser.parse_args()
    if not 0 < args.preferred_seconds <= args.maximum_seconds <= 30:
        raise ValueError("require 0 < preferred-seconds <= maximum-seconds <= 30")
    if not 0 <= args.overlap_seconds < args.preferred_seconds:
        raise ValueError("overlap must be non-negative and shorter than preferred length")
    audio, rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz input, got {rate}; resample before planning")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    intervals = speech_intervals(audio, args.vad_threshold_db, args.frame_ms,
                                 args.min_speech_ms, args.merge_silence_ms)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Segment] = []
    for start, end in intervals:
        for left, right, forced in split_interval(start, end, args.preferred_seconds,
                                                   args.maximum_seconds, args.overlap_seconds):
            profile, bucket = choose_profile((right - left) / SAMPLE_RATE)
            segments.append(Segment(len(segments), left, right, profile, bucket, forced))
    for segment in segments:
        name = f"segment_{segment.index:05d}_{segment.profile}.wav"
        clip = audio[segment.start_sample:segment.end_sample]
        target_samples = round(segment.bucket_seconds * SAMPLE_RATE)
        if len(clip) < target_samples:
            clip = np.pad(clip, (0, target_samples - len(clip)))
        sf.write(args.output_dir / name, clip[:target_samples], SAMPLE_RATE)
    payload = {
        "version": 1,
        "audio": str(args.audio),
        "sample_rate": SAMPLE_RATE,
        "vad": {"implementation": "energy", "threshold_db": args.vad_threshold_db,
                "frame_ms": args.frame_ms, "min_speech_ms": args.min_speech_ms,
                "merge_silence_ms": args.merge_silence_ms},
        "chunking": {"preferred_seconds": args.preferred_seconds, "maximum_seconds": args.maximum_seconds,
                     "overlap_seconds": args.overlap_seconds},
        "segments": [segment.json() for segment in segments],
    }
    (args.output_dir / "plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"segments": len(segments), "speech_seconds": round(sum(s.end_sample-s.start_sample for s in segments)/SAMPLE_RATE, 3), "plan": str(args.output_dir/'plan.json')}, indent=2))


if __name__ == "__main__":
    main()
