"""Create a deterministic, labelled five-item ASR smoke-test corpus."""

import argparse
import json
from pathlib import Path

import soundfile as sf
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--pad-to-seconds", type=float)
    args = parser.parse_args()

    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    output = args.output
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(dataset.select(range(args.count))):
        audio = item["audio"]
        samples = audio["array"]
        if args.pad_to_seconds is not None:
            target_samples = int(round(args.pad_to_seconds * audio["sampling_rate"]))
            if len(samples) > target_samples:
                raise ValueError(f"sample {item['id']} exceeds requested fixed duration")
            samples = __import__("numpy").pad(samples, (0, target_samples - len(samples)))
        path = audio_dir / f"{index:02d}_{item['id']}.wav"
        sf.write(path, samples, audio["sampling_rate"], subtype="PCM_16")
        rows.append(
            {
                "index": index,
                "id": item["id"],
                "audio": str(path.resolve()),
                "reference": item["text"],
                "sampling_rate": int(audio["sampling_rate"]),
                "speech_duration_s": len(audio["array"]) / float(audio["sampling_rate"]),
                "duration_s": len(samples) / float(audio["sampling_rate"]),
            }
        )
    (output / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
