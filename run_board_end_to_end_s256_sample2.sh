#!/usr/bin/env bash
set -euo pipefail
remote=root@192.168.2.27:/root/audio8-asr/decoder_3d_fixed_s256/
sshpass -p 'yujiarong520' scp -q /root/audio8-asr/prefill_kv_s110_s2/input_features.npy /root/audio8-asr/prefill_kv_s110_s2/input_ids.npy /root/audio8-asr/prefill_kv_s110_s2/audio_positions.npy /root/audio8-asr/prefill_kv_s110_s2/rotary_inv_freq.npy "$remote"
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'cd /root/audio8-asr/decoder_3d_fixed_s256 && python3 board_run_end_to_end_rknn.py --root . --model /root/audio8-asr/model --max-new 128 --output end_to_end_s2.json'
