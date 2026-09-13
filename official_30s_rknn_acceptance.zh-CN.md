# Audio8-ASR 官方 30 秒输入路径：RK3576 FP16 RKNN 验收

## 对齐契约

本测试严格采用上游 README 的公开示例契约，而不是此前 F800 的 8 秒实验参数：

```python
audio_padding = "longest"
audio_max_length = 30 * 16000
sampling_rate = 16000
max_new_tokens = 128
do_sample = False
```

测试输入为精确 30.000 秒、16 kHz、单声道 WAV。前半来自公开测试音频，尾部零补齐，只用于让输入满足官方 cap 的精确边界；它不用于对外 WER 基准结论。

原版 processor 的实际输出为：mel `[128,3000]`、375 个 `<|audio|>` placeholder，prompt/prefill 序列长度为 385。

## 图与原版等价性

| 阶段 | RKNN 图 | 静态输入/输出 | 对齐要点 |
|---|---|---|---|
| Audio encoder | `audio_encoder_f3000_fp16.rknn` | `[128,3000] → [390,1024]` | 严格复刻 `n_window_infer=800` 的 `cu_seqlens=[0,104,208,312,390]`；导出前 PyTorch 最大绝对误差为 0.0 |
| Adapter/projector | `audio_adapter_h390_t375_fp16.rknn` | `[390,1024] → [375,512]` | 严格按原始 adaptive pooling 输出 375 个 embedding |
| Prefill | `prefill_kv_s385_fp16.rknn` | `[1,385,512] → logits + 8 层 K/V` | 输出初始 K/V `[1,8,385,64]` |
| Decoder | `block_s512` | rank-3 CPU cache `[8,512,64]` | S385 自动选择 S512 作为初始容量 |

早期 F3000 静态 encoder 曾错误地使用一个 `[0,390]` attention window；它对 F800 恰好无影响，却会让 30 秒输出偏离原版（max abs 0.11546）。现已修正为四个原版窗口，严格 PyTorch 导出等价检验为 0.0。

## 端到端结果

| 项目 | 结果 |
|---|---:|
| 原始 FP32 token（含 EOS） | 85 |
| RKNN token（含 EOS） | 85 |
| 全 token 对比 | **85/85 完全一致** |
| 首个 token 分歧 | 无 |
| EOS | 两端均为 `151645` |
| NPU vs FP32 CER/WER | **0% / 0%** |
| 初始 KV bucket | 512 |
| Cache bucket 切换 | 无；EOS 前 context 未超过 512 |

## RK3576 性能

| 项目 | 时延 |
|---|---:|
| Encoder NPU 调用 | 1603.81 ms |
| Adapter NPU 调用 | 180.33 ms |
| Prefill NPU 调用 | 477.05 ms |
| Decoder NPU 调用总计 | 19835.14 ms |
| CPU K/V 原地写入总计 | 24.99 ms |
| 稳态 neural E2E | **22121.31 ms** |
| RTF | **0.737** |
| 包含模型加载的 process wall | 32459.20 ms |
| 结束 RSS | 1206.16 MiB |

RKNN Lite 的 NPU 调用时延包含 host buffer 提交、NPU 执行与同步等待；本次未使用 zero-copy API，故不虚构单独的 H2D/D2H 时间。
