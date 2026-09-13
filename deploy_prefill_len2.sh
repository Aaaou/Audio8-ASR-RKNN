#!/usr/bin/env bash
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'rm -rf /root/audio8-asr/prefill_input_s2; mkdir -p /root/audio8-asr/prefill_input_s2'
sshpass -p 'yujiarong520' scp -q /root/audio8-asr/prefill_input_s2/prefill_kv_s35_fp16.rknn /root/audio8-asr/prefill_input_s2/input_features.npy /root/audio8-asr/prefill_input_s2/input_ids.npy /root/audio8-asr/prefill_input_s2/audio_positions.npy /root/audio8-asr/prefill_input_s2/rotary_inv_freq.npy root@192.168.2.27:/root/audio8-asr/prefill_input_s2/
