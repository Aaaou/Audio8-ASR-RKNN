# Audio8-ASR RKNN deployment archive

This repository records the RK3576 deployment work for
[Edge0/Audio8-ASR-0.1B](https://huggingface.co/Edge0/Audio8-ASR-0.1B).
It contains export/build tools, board runtimes, acceptance evidence, and the
dynamic CPU-owned KV-cache scheduler.  It deliberately excludes model weights,
audio samples, ONNX intermediates, NumPy tensor captures, and `.rknn` binaries.
The deployable RKNN release is published separately on Hugging Face.

## Verified architecture

The model is an autoregressive ASR system:

```text
audio features
  → audio encoder (RKNN/NPU)
  → audio MLP tower + projector (RKNN/NPU)
  → Qwen2 prefill (RKNN/NPU; logits + initial K/V)
  → Qwen2 token decode loop (RKNN/NPU)
  → EOS
```

The decoder uses eight Qwen2 layers with `hidden_size=512`, eight attention/KV
heads, `head_dim=64`, and an FP32 host-owned external cache of shape
`[heads, Smax, head_dim]`.  NPU graphs are stateless: CPU manages only cache
buffers, positions, masks, RoPE inputs, embedding lookup and argmax.

Static RKNN cache capacities grow `128 → 256 → 512`; valid cache contents are
copied on the CPU and the next block graph is selected without replaying the
whole prompt.  Do not use the older graph-internal `DynamicCache`/Concat route:
it was shown inaccurate on RK3576.

## Acceptance

The principal report is
[dynamic_bucket_multi_duration_acceptance.zh-CN.md](dynamic_bucket_multi_duration_acceptance.zh-CN.md).
It records real EOS tests at 2/4/6/8 seconds, all token-identical to original
PyTorch CPU FP32 (`CER=0%`, `WER=0%`), plus an EOS-ignored cache stress run that
actually verifies `128→256→512` against 181 FP32 reference tokens.

## Important entry points

- `board_run_end_to_end_dynamic_buckets.py` — RK3576 full neural runtime.
- `export_qwen2_prefill_with_kv.py` — fixed-shape prefill export.
- `export_decoder_3d_fixed_cache.py` — rank-3 external-cache decoder blocks.
- `export_audio_encoder_bucket.py`, `export_audio_adapter_bucket.py` — frontend
  static bucket exports, including the T25/T50/T75/T100 adapter variants.
- `generate_length_standard_reference.py` and
  `generate_forced_decoder_reference.py` — deterministic FP32 references.

## Scope and licensing

The upstream Audio8 model and its weights remain subject to their upstream
license and terms.  No upstream model weights are committed here.  The release
instructions require obtaining the original model snapshot separately.
