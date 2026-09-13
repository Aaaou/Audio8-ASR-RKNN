#!/usr/bin/env bash
set -euo pipefail
remote=root@192.168.2.27:/root/audio8-asr/prefill_kv_s110/
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'mkdir -p /root/audio8-asr/prefill_kv_s110'
sshpass -p 'yujiarong520' scp -q /root/audio8-asr/prefill_kv_s110/prefill_kv_s110_fp16.rknn /root/audio8-asr/prefill_kv_s110/*.npy /mnt/c/Users/Administrator/Documents/ChatGPT/asr/board_test_prefill_kv.py "$remote"
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'cd /root/audio8-asr/prefill_kv_s110 && python3 board_test_prefill_kv.py --dir . --output board_prefill_kv.json'
