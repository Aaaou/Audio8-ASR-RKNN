"""Compare RKNN long-audio tasks to same-contract FP32 references and stitch text."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def edit_distance(left: list[str], right: list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, item in enumerate(left, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(right, 1):
            old = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (item != other))
            previous = old
    return row[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--rknn", type=Path, required=True)
    parser.add_argument("--fp32-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))["segments"]
    board = {int(item["index"]): item for item in json.loads(args.rknn.read_text(encoding="utf-8"))["tasks"]}
    records, texts = [], []
    for segment in plan:
        index = int(segment["index"])
        reference = json.loads((args.fp32_dir / f"segment_{index:05d}.json").read_text(encoding="utf-8"))
        actual = board[index]
        same = reference["tokens"] == actual["tokens"]
        records.append({"index": index, "profile": segment["profile"], "start_ms": segment["start_ms"],
                        "end_ms": segment["end_ms"], "tokens_equal": same,
                        "eos_equal": reference["complete_to_eos"] == actual["complete_to_eos"],
                        "token_count": len(actual["tokens"]), "text": reference["text"],
                        "neural_ms": actual["latency_ms"]["neural_total"],
                        "task_wall_ms": actual["latency_ms"]["task_wall_total"],
                        "rss_mib": actual.get("rss_mib", {})})
        texts.append(reference["text"])
    fp_text, rknn_text = " ".join(texts), " ".join(texts) if all(x["tokens_equal"] for x in records) else ""
    words_left, words_right = normalized(fp_text).split(), normalized(rknn_text).split()
    chars_left, chars_right = list(normalized(fp_text).replace(" ", "")), list(normalized(rknn_text).replace(" ", ""))
    payload = {"acceptance": {"all_tokens_equal": all(x["tokens_equal"] for x in records),
               "all_eos_equal": all(x["eos_equal"] for x in records),
               "wer_percent": 100 * edit_distance(words_left, words_right) / max(1, len(words_left)),
               "cer_percent": 100 * edit_distance(chars_left, chars_right) / max(1, len(chars_left))},
               "summary": {"segments": len(records), "audio_span_ms": plan[-1]["end_ms"] - plan[0]["start_ms"],
               "neural_total_ms": sum(x["neural_ms"] for x in records),
               "task_wall_total_ms": sum(x["task_wall_ms"] for x in records), "text": fp_text}, "segments": records}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
