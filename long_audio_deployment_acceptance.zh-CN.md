# Audio8-ASR RK3576 长音频调度验收

## 结论

已落地可复现的应用层路径：**VAD → 8/16/30 秒有限前端 bucket → 每段独立 prefill/KV → EOS → 按时间轴拼接**。RK3576 上的真实 30 秒体验输入被 VAD 分成 5 段（4 个 s8，1 个 s16）；五段均 EOS，且相同 `max_length` 输入契约下，RKNN 与原始 PyTorch CPU FP32 均逐 token 相同，故 NPU-vs-FP32 **CER=0%、WER=0%**。

这不是带人工标注的绝对 WER benchmark；它验证的是长音频调度层没有相对 FP32 引入数值或控制流回归。

## 组件和语义

| 文件 | 作用 |
|---|---|
| `long_audio_segmenter.py` | 16 kHz mono WAV 的 energy-VAD、段合并、超长段 overlap 切分、profile 分配和波形级补零。 |
| `prepare_deployment_bucket_inputs.py` | 由原版 processor 根据 profile 生成固定 padding 的模型输入。 |
| `board_run_long_audio.py` | RK3576 worker：每段独立 prefill/KV/EOS，统计时延、RSS、KV bucket。 |
| `compare_long_audio_acceptance.py` | 对相同 profile 的 FP32 与 RKNN JSON 进行 token/EOS/CER/WER 验收。 |

每个段的 cache 在 EOS 后被丢弃，绝不跨音频段传递。这符合 Audio8 当前离线自回归 ASR 的模型语义；它不是原生 cross-chunk streaming checkpoint。

## 新增 s16 图

| 阶段 | 图 | shape |
|---|---|---|
| Encoder | `audio_encoder_f1600_fp16.rknn` | `[128,1600] → [208,1024]` |
| Adapter | `audio_adapter_h208_t200_fp16.rknn` | `[208,1024] → [200,512]` |
| Prefill | `prefill_kv_s210_fp16.rknn` | `[1,210,512] → logits + 8 层初始 K/V` |

F1600 encoder 按上游 `n_window_infer=800` 保留两个 attention window，而不是错误地把 208 个 token 合为一个 window。

## 30 秒真实体验与 BM

输入为精确 30 秒测试音频。energy-VAD 使用 -60 dB、30 ms frame、250 ms 最小有声、300 ms 静音合并；分出五个自然边界段，本轮没有 forced overlap。FP32 与 RKNN 均采用 `audio_padding="max_length"`，且波形已补零到所选 profile。

| 段 | 范围 | profile | token（含 EOS） | NPU+CPU neural | task wall | KV | FP32 对齐 |
|---:|---:|---|---:|---:|---:|---|---|
| 0 | 0.00–2.52 s | s8 | 10 | 3494.75 ms | 10304.41 ms | S128 | 全同 |
| 1 | 3.21–4.95 s | s8 | 7 | 2955.81 ms | 9509.00 ms | S128 | 全同 |
| 2 | 5.40–10.08 s | s8 | 16 | 4585.77 ms | 11095.83 ms | S128 | 全同 |
| 3 | 10.77–18.63 s | s8 | 27 | 8547.55 ms | 15078.15 ms | S128→S256 | 全同 |
| 4 | 19.35–29.19 s | s16 | 33 | 8810.02 ms | 16428.76 ms | S256 | 全同 |
| **总计** | **原文件 30.00 s** | — | **93** | **28393.90 ms** | **62416.16 ms** | — | **CER/WER=0%/0%** |

`neural` 包含 encoder、adapter、prefill、decoder NPU 调用和 CPU KV 写入，RTF=`28.394/30`=**0.946**。`task wall` 还含 RKNN 图映射/初始化/释放，RTF=**2.081**；不含主机侧 VAD、音频解码和 processor 准备。

第 3 段的 `S128→S256` CPU KV copy 为 **38.23 ms**，已在原始 JSON 中独立记录；不是 CPU 重算 decoder。

## 内存结论

让所有前端 profile 和 decoder 图同时常驻会触发 4 GiB RK3576 的 OOM。正式 worker 因此强制分阶段映射：

```text
encoder + adapter + prefill → 释放 frontend 图 → decoder 图 → EOS
```

本轮最大 RSS：frontend stage **1662.31 MiB**，decoder stage **1068.11 MiB**。运行日志显示所有图 target 是 `RKNPU f2`；这只是图映射生命周期管理，不是 CPU fallback。

## 体验文本

输出全程非空且正常 EOS：

> Linil's pictures are a sort of. Guards and Adam paintings. And Mason's exquisite idles are as national as a jingo poem. Mr. Barkett Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mister John Collier gives his sitter a cheerful slap on the back. Before he says, like a shampooer in a Turkish bath, next man.

没有人工参考文本，故它只作为段间连续性、文本非空、EOS 与 FP32/RKNN 一致性的体验验证。

## 复现

```bash
python long_audio_segmenter.py --audio INPUT.wav --output-dir plan
python prepare_deployment_bucket_inputs.py --model MODEL_DIR --plan-dir plan --output-dir inputs
python3 board_run_long_audio.py --asset-root RELEASE_ROOT --inputs-manifest inputs/inputs-manifest.json --token-embeddings RELEASE_ROOT/token_embeddings_fp32.npy --output long_audio_rknn.json
python compare_long_audio_acceptance.py --plan plan/plan.json --rknn long_audio_rknn.json --fp32-dir fp32 --output acceptance.json
```

后续验收：强制超长段的 overlap 去重、S1024 decoder，以及带人工标注的长音频绝对 WER/CER。
