#!/usr/bin/env bash
set -euo pipefail
remote=root@192.168.2.27:/root/audio8-asr/decoder_3d_fixed_s128/
sshpass -p 'yujiarong520' scp -q -r /root/audio8-asr/decoder_3d_fixed_s128/schedule_refs "$remote"
sshpass -p 'yujiarong520' scp -q /mnt/c/Users/Administrator/Documents/ChatGPT/asr/board_test_decoder_3d_fixed_schedule.py "$remote"
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'cd /root/audio8-asr/decoder_3d_fixed_s128 && python3 board_test_decoder_3d_fixed_schedule.py --dir . --output board_schedule_6step.json'
