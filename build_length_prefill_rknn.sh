#!/usr/bin/env bash
set -euo pipefail
source /root/.venv-rknn232/bin/activate
python /mnt/c/Users/Administrator/Documents/ChatGPT/asr/build_rknn_fp16.py --onnx /root/audio8-asr/prefill_input_s2/prefill_kv_s35.onnx --output /root/audio8-asr/prefill_input_s2/prefill_kv_s35_fp16.rknn >/tmp/build_prefill_s2.log 2>&1
echo s2_done
python /mnt/c/Users/Administrator/Documents/ChatGPT/asr/build_rknn_fp16.py --onnx /root/audio8-asr/prefill_input_s4/prefill_kv_s60.onnx --output /root/audio8-asr/prefill_input_s4/prefill_kv_s60_fp16.rknn >/tmp/build_prefill_s4.log 2>&1
echo s4_done
python /mnt/c/Users/Administrator/Documents/ChatGPT/asr/build_rknn_fp16.py --onnx /root/audio8-asr/prefill_input_s6/prefill_kv_s85.onnx --output /root/audio8-asr/prefill_input_s6/prefill_kv_s85_fp16.rknn >/tmp/build_prefill_s6.log 2>&1
echo s6_done
