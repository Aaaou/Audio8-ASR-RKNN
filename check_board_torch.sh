#!/usr/bin/env bash
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'python3 -c "import torch; print(torch.__version__)"; find /root/audio8-asr/model -maxdepth 1 -name "*.safetensors" -printf "%f\n"; python3 -c "from safetensors import safe_open; p=\"/root/audio8-asr/model/model.safetensors\"; f=safe_open(p,framework=\"pt\"); print([x for x in f.keys() if \"embed_tokens\" in x])"'
