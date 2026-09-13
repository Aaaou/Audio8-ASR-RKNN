#!/usr/bin/env bash
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'find /root/audio8-asr/decoder_3d_fixed_s128 -name "*.rknn" | wc -l; test -f /root/audio8-asr/decoder_3d_fixed_s128/board_one_step.json && cat /root/audio8-asr/decoder_3d_fixed_s128/board_one_step.json || true'
