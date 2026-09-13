#!/usr/bin/env bash
set -euo pipefail
remote=root@192.168.2.27:/root/audio8-asr/decoder_3d_fixed_s256/
sshpass -p 'yujiarong520' scp -q /root/audio8-asr/prefill_kv_s110/input_features.npy /root/audio8-asr/prefill_kv_s110/input_ids.npy /root/audio8-asr/prefill_kv_s110/audio_positions.npy /root/audio8-asr/prefill_kv_s110/rotary_inv_freq.npy root@192.168.2.27:/root/audio8-asr/prefill_kv_s110/
sshpass -p 'yujiarong520' scp -q /mnt/c/Users/Administrator/Documents/ChatGPT/asr/board_run_end_to_end_dynamic_buckets.py "$remote"
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'cd /root/audio8-asr/decoder_3d_fixed_s256 && python3 board_run_end_to_end_dynamic_buckets.py --input-dir /root/audio8-asr/prefill_kv_s110 --model /root/audio8-asr/model --output dynamic_s1.json'
