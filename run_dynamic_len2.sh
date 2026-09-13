#!/usr/bin/env bash
sshpass -p 'yujiarong520' scp -q /mnt/c/Users/Administrator/Documents/ChatGPT/asr/board_run_end_to_end_dynamic_buckets.py root@192.168.2.27:/root/audio8-asr/decoder_3d_fixed_s256/
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'cd /root/audio8-asr/decoder_3d_fixed_s256 && python3 board_run_end_to_end_dynamic_buckets.py --input-dir /root/audio8-asr/prefill_input_s2 --prefill-rknn /root/audio8-asr/prefill_input_s2/prefill_kv_s35_fp16.rknn --model /root/audio8-asr/model --output dynamic_length_s2.json'
