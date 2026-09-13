---
license: other
library_name: rknn
tags:
  - automatic-speech-recognition
  - audio8-asr
  - rknn
  - rk3576
  - npu
  - qwen2
  - fp16
base_model: Edge0/Audio8-ASR-0.1B
---

# Audio8-ASR-0.1B — RK3576 FP16 RKNN deployment

This repository provides a deployable FP16 RKNN partition of
[Edge0/Audio8-ASR-0.1B](https://huggingface.co/Edge0/Audio8-ASR-0.1B) for the
Rockchip RK3576 NPU, plus the board runtime needed to drive it.

It is **not** a replacement for the upstream model snapshot.  Download the
upstream model separately for `config.json`, tokenizer/processor files and the
original weights used by host-side preprocessing and validation.  Follow the
upstream model license and terms in addition to this release's files.

## What is included

The release layout is intentionally explicit:

```text
audio8-asr-rknn-rk3576/
├── encoder/audio_encoder_f800_fp16.rknn
├── adapter/
│   ├── audio_adapter_h104_t25_fp16.rknn
│   ├── audio_adapter_h104_t50_fp16.rknn
│   ├── audio_adapter_h104_t75_fp16.rknn
│   └── audio_adapter_h104_t100_fp16.rknn
├── prefill/
│   ├── prefill_kv_s35_fp16.rknn
│   ├── prefill_kv_s60_fp16.rknn
│   ├── prefill_kv_s85_fp16.rknn
│   └── prefill_kv_s110_fp16.rknn
├── decoder/
│   ├── kv/layer0..layer7/kv_fp16.rknn
│   ├── block_s128/layer0..layer7/block_fp16.rknn
│   ├── block_s256/layer0..layer7/block_fp16.rknn
│   ├── block_s512/layer0..layer7/block_fp16.rknn
│   └── head_shards/shard00..shard07/head_fp16.rknn
├── token_embeddings_fp32.npy
├── runtime/board_run_end_to_end_dynamic_buckets.py
├── runtime/requirements-rk3576.txt
├── docs/benchmark-and-acceptance.zh-CN.md
└── docs/architecture-and-operation.zh-CN.md
```

The release does not include upstream weights, source audio, ONNX intermediates,
or calibration/validation tensors.  `token_embeddings_fp32.npy` is the decoder
embedding lookup table exported from the upstream snapshot; it is included
because the lightweight board runtime uses it for each generated token.

## Architecture

Audio8-ASR is autoregressive ASR.  The runtime sequence is:

```text
mel features (CPU preprocessing)
  → audio encoder                           [RKNN / NPU]
  → audio MLP tower + projector             [RKNN / NPU]
  → Qwen2 prefill, logits + initial K/V     [RKNN / NPU]
  → repeated Qwen2 token decoding to EOS    [RKNN / NPU]
```

The language decoder has 8 layers, hidden size 512, 8 attention/KV heads and
head dimension 64.  Its K/V cache is deliberately **host-owned**, rank-3 FP32
buffers `[heads, Smax, head_dim] = [8, Smax, 64]`.  The CPU performs only cache
allocation/updates, RoPE and attention-mask preparation, embedding lookup and
argmax.  All neural layers stay in RKNN NPU graphs.

RKNN uses static shapes, while the upstream model uses a DynamicCache.  The
runtime reproduces the behavior through cache capacities `128 → 256 → 512`.
When full, it copies only the valid K/V region to a larger CPU buffer and picks
the matching stateless NPU decoder-block graph.  There is no graph-internal
Concat, ScatterND or runtime-owned persistent cache.

## Runtime requirements

Tested target:

```text
RK3576
RKNPU driver 0.9.8
RKNN Lite2 / Runtime 2.3.2
Python 3 with numpy and rknnlite.api
```

Install the `runtime/` directory and the model tree at the same deployment root
used in its constants, or change those paths in the runtime.  The runtime takes
already-prepared `input_features.npy`, `input_ids.npy`, `audio_positions.npy`
and `rotary_inv_freq.npy`; generate them using the accompanying source archive
with the same upstream processor and prompt template.

For each audio length, select matching static components:

| Audio duration | audio placeholders | Adapter | Prefill graph |
|---:|---:|---|---|
| 2 s | 25 | T25 | S35 |
| 4 s | 50 | T50 | S60 |
| 6 s | 75 | T75 | S85 |
| 8 s | 100 | T100 | S110 |

The adapter output count must exactly equal the audio-placeholder count.  Do
not take the first N values from T100 for a shorter prompt: that changes the
upstream adaptive-pooling semantics and is incorrect.

Example board invocation after preparing an S85 input:

```bash
python3 runtime/board_run_end_to_end_dynamic_buckets.py \
  --input-dir prefill_input_s6 \
  --prefill-rknn prefill/prefill_kv_s85_fp16.rknn \
  --adapter-rknn adapter/audio_adapter_h104_t75_fp16.rknn \
  --model /path/to/upstream-Audio8-ASR-0.1B \
  --output result.json
```

The program stops at the upstream EOS id under normal operation.  Its
`--ignore-eos` option is only for cache scheduler stress validation; do not use
it for user-facing transcription.

## Validated accuracy and board benchmark

All regular tests ran from real RKNN prefill through real RKNN autoregressive
decoding to EOS on RK3576.  Each output was token-identical to the original
PyTorch CPU FP32 greedy reference; thus NPU-vs-FP32 CER/WER was 0%/0%.

| Audio | Prefill length | KV capacity used | Steady neural E2E | RTF | End RSS |
|---:|---:|---|---:|---:|---:|
| 2 s | 35 | 128 | 1231.49 ms | 0.616 | 1113.15 MiB |
| 4 s | 60 | 128 | 3371.87 ms | 0.843 | 1114.12 MiB |
| 6 s | 85 | 128 | 4116.86 ms | 0.686 | 1116.10 MiB |
| 8 s | 110 | 128 → 256 | 4344.29 ms | 0.543 | 1232.23 MiB |

`Steady neural E2E` includes encoder, adapter, prefill, decoder RKNN calls,
CPU K/V writes and any cache migration.  It excludes model loading/runtime
initialization.  Normal RKNN Lite `inference()` timing combines buffer transfer,
NPU work and synchronization; the presented numbers do not falsely claim a
separate DMA/H2D/D2H time.

Additional EOS-ignored scheduler stress testing ran 180 decoder steps from a
real 6-second prefill, triggering both `128→256` and `256→512`.  All 181 NPU
tokens matched FP32; cache copies cost 11.21 ms and 34.70 ms respectively, and
the final RSS was 1379.02 MiB.  This validates cache scheduling, not ASR text
quality beyond EOS.

## Reproducibility source archive

Export scripts, input preparation code, board probes, JSON evidence and full
Chinese acceptance documentation are in the companion GitHub branch:

https://github.com/Aaaou/Audio8-ASR-RKNN/tree/rknn-dynamic-kv-cache-validated
