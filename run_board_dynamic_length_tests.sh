#!/usr/bin/env bash
set -euo pipefail
for n in 2 4 6; do
  remote=/root/audio8-asr/dynamic_input_s${n}
  sshpass -p 'yujiarong520' ssh root@192.168.2.27 "rm -rf '$remote'; mkdir -p '$remote'"
  sshpass -p 'yujiarong520' scp -q /root/audio8-asr/prefill_input_s${n}/input_features.npy /root/audio8-asr/prefill_input_s${n}/input_ids.npy /root/audio8-asr/prefill_input_s${n}/audio_positions.npy /root/audio8-asr/prefill_input_s${n}/rotary_inv_freq.npy root@192.168.2.27:"$remote/"
  sshpass -p 'yujiarong520' ssh root@192.168.2.27 "cd /root/audio8-asr/decoder_3d_fixed_s256 && python3 board_run_end_to_end_dynamic_buckets.py --input-dir '$remote' --model /root/audio8-asr/model --output dynamic_length_s${n}.json"
done
