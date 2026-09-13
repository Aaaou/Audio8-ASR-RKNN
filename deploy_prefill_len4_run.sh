#!/usr/bin/env bash
set -euo pipefail
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'rm -rf /root/audio8-asr/prefill_input_s4; mkdir -p /root/audio8-asr/prefill_input_s4'
sshpass -p 'yujiarong520' scp -q /root/audio8-asr/prefill_input_s4/prefill_kv_s60_fp16.rknn /root/audio8-asr/prefill_input_s4/input_features.npy /root/audio8-asr/prefill_input_s4/input_ids.npy /root/audio8-asr/prefill_input_s4/audio_positions.npy /root/audio8-asr/prefill_input_s4/rotary_inv_freq.npy root@192.168.2.27:/root/audio8-asr/prefill_input_s4/
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'cd /root/audio8-asr/decoder_3d_fixed_s256 && python3 board_run_end_to_end_dynamic_buckets.py --input-dir /root/audio8-asr/prefill_input_s4 --prefill-rknn /root/audio8-asr/prefill_input_s4/prefill_kv_s60_fp16.rknn --model /root/audio8-asr/model --output dynamic_length_s4.json'
