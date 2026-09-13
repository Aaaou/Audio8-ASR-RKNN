#!/usr/bin/env bash
set -euo pipefail
remote=root@192.168.2.27:/root/audio8-asr/prefill_kv_s110/
sshpass -p 'yujiarong520' scp -q /root/audio8-asr/prefill_kv_s110/input_embeddings.npy /root/audio8-asr/prefill_kv_s110/input_features.npy /root/audio8-asr/prefill_kv_s110/audio_positions.npy /mnt/c/Users/Administrator/Documents/ChatGPT/asr/board_test_frontend_prefill_kv.py "$remote"
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'cd /root/audio8-asr/prefill_kv_s110 && python3 board_test_frontend_prefill_kv.py --dir . --output board_frontend_prefill_kv.json'
