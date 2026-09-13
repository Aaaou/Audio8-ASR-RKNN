#!/usr/bin/env bash
set -euo pipefail
remote=root@192.168.2.27:/root/audio8-asr/decoder_3d_fixed_s256/
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'rm -rf /root/audio8-asr/decoder_3d_fixed_s256; mkdir -p /root/audio8-asr/decoder_3d_fixed_s256'
sshpass -p 'yujiarong520' scp -q -r /root/audio8-asr/decoder_3d_fixed_s256/layer{0..7} "$remote"
sshpass -p 'yujiarong520' scp -q /root/audio8-asr/prefill_kv_s110/input_features.npy /root/audio8-asr/prefill_kv_s110/input_ids.npy /root/audio8-asr/prefill_kv_s110/audio_positions.npy /root/audio8-asr/prefill_kv_s110/rotary_inv_freq.npy /root/audio8-asr/qwen_embed_fp32.npy /mnt/c/Users/Administrator/Documents/ChatGPT/asr/board_run_end_to_end_rknn.py "$remote"
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'mv /root/audio8-asr/decoder_3d_fixed_s256/qwen_embed_fp32.npy /root/audio8-asr/decoder_3d_fixed_s256/token_embeddings_fp32.npy'
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'cd /root/audio8-asr/decoder_3d_fixed_s256 && python3 board_run_end_to_end_rknn.py --root . --model /root/audio8-asr/model --max-new 128 --output end_to_end.json'
