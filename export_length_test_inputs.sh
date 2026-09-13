#!/usr/bin/env bash
set -euo pipefail
source /root/.venv-rknn232/bin/activate
for n in 2 4 6; do
  out="/root/audio8-asr/prefill_input_s${n}"
  rm -rf "$out"
  python /mnt/c/Users/Administrator/Documents/ChatGPT/asr/export_qwen2_prefill_with_kv.py --model /root/audio8-asr/model --audio "/root/audio8-asr/length_test_audio/s${n}.wav" --output "$out"
done
