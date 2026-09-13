#!/usr/bin/env bash
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'ls -lh /root/audio8-asr/decoder_3d_fixed_s256/token_embeddings_fp32.npy 2>/dev/null; cat /root/audio8-asr/decoder_3d_fixed_s256/end_to_end.json 2>/dev/null || true'
