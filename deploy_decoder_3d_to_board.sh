#!/usr/bin/env bash
set -euo pipefail
remote=root@192.168.2.27
remote_dir=/root/audio8-asr/decoder_3d_fixed_s128
sshpass -p 'yujiarong520' ssh -o StrictHostKeyChecking=no "$remote" "rm -rf '$remote_dir'; mkdir -p '$remote_dir'"
sshpass -p 'yujiarong520' scp -q -r /root/audio8-asr/decoder_3d_fixed_s128/layer{0..7} /root/audio8-asr/decoder_3d_fixed_s128/head /root/audio8-asr/decoder_3d_fixed_s128/head_shards "$remote:$remote_dir/"
sshpass -p 'yujiarong520' scp -q /mnt/c/Users/Administrator/Documents/ChatGPT/asr/board_test_decoder_3d_fixed_step.py "$remote:$remote_dir/"
sshpass -p 'yujiarong520' ssh "$remote" "cd '$remote_dir' && python3 board_test_decoder_3d_fixed_step.py --dir . --output board_one_step.json"
