#!/usr/bin/env bash
set -euo pipefail
source /root/.venv-rknn232/bin/activate
cd /root/audio8-asr/decoder_3d_fixed_s128
for d in head_shards/shard{02..07}; do
  python /mnt/c/Users/Administrator/Documents/ChatGPT/asr/build_rknn_fp16.py --onnx "$d/head.onnx" --output "$d/head_fp16.rknn" >"/tmp/build_${d//\//_}.log" 2>&1
  echo "$d done"
done
