#!/usr/bin/env bash
# Stage release binaries from the verified WSL build tree.  It intentionally
# copies no upstream weights, audio, ONNX files or tensor captures.
set -euo pipefail
src=/root/audio8-asr
dst=${1:?usage: stage_release_from_wsl.sh DESTINATION}
mkdir -p "$dst"/{encoder,adapter,prefill,decoder/kv,decoder/block_s128,decoder/block_s256,decoder/block_s512,decoder/head_shards,runtime,docs}
cp "$src/encoder_f800_v2/audio_encoder_f800_fp16.rknn" "$dst/encoder/"
cp "$src/encoder_f3000_official30/audio_encoder_f3000_fp16.rknn" "$dst/encoder/"
cp "$src/decoder_3d_fixed_s256/token_embeddings_fp32.npy" "$dst/"
cp "$src"/audio_adapter_h104_t{25,50,75,100}/audio_adapter_h104_t*_fp16.rknn "$dst/adapter/"
cp "$src/audio_adapter_h390_t375_official30/audio_adapter_h390_t375_fp16.rknn" "$dst/adapter/"
cp "$src/prefill_input_s2/prefill_kv_s35_fp16.rknn" "$dst/prefill/"
cp "$src/prefill_input_s4/prefill_kv_s60_fp16.rknn" "$dst/prefill/"
cp "$src/prefill_input_s6/prefill_kv_s85_fp16.rknn" "$dst/prefill/"
cp "$src/prefill_kv_s110/prefill_kv_s110_fp16.rknn" "$dst/prefill/"
cp "$src/prefill_input_official30/prefill_kv_s385_fp16.rknn" "$dst/prefill/"
for n in 0 1 2 3 4 5 6 7; do
  mkdir -p "$dst/decoder/kv/layer$n" "$dst/decoder/block_s128/layer$n" "$dst/decoder/block_s256/layer$n" "$dst/decoder/block_s512/layer$n"
  cp "$src/decoder_3d_fixed_s128/layer$n/kv_fp16.rknn" "$dst/decoder/kv/layer$n/"
  cp "$src/decoder_3d_fixed_s128/layer$n/block_fp16.rknn" "$dst/decoder/block_s128/layer$n/"
  cp "$src/decoder_3d_fixed_s256/layer$n/block_fp16.rknn" "$dst/decoder/block_s256/layer$n/"
  cp "$src/decoder_3d_fixed_s512/layer$n/block_fp16.rknn" "$dst/decoder/block_s512/layer$n/"
done
for n in 00 01 02 03 04 05 06 07; do mkdir -p "$dst/decoder/head_shards/shard$n"; cp "$src/decoder_3d_fixed_s128/head_shards/shard$n/head_fp16.rknn" "$dst/decoder/head_shards/shard$n/"; done
echo "Staged binaries in $dst. Copy runtime/docs from this repository before upload."
