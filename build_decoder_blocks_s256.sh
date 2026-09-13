#!/usr/bin/env bash
set -euo pipefail
source /root/.venv-rknn232/bin/activate
cd /root/audio8-asr/decoder_3d_fixed_s256
for d in layer{0..7}; do
  # K/V projection is shape-invariant; only attention blocks depend on Smax.
  cp /root/audio8-asr/decoder_3d_fixed_s128/"$d"/kv_fp16.rknn "$d"/kv_fp16.rknn
  python /mnt/c/Users/Administrator/Documents/ChatGPT/asr/build_rknn_fp16.py --onnx "$d/block.onnx" --output "$d/block_fp16.rknn" >"/tmp/build_s256_${d}.log" 2>&1
  echo "$d done"
done
