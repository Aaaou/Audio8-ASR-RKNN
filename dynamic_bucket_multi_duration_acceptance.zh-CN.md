# Audio8-ASR：动态 KV-cache bucket 与多时长端到端验收

**结论：通过本阶段验收。** 在 RK3576 上，FP16 RKNN 路径以 CPU 持有 KV buffer、NPU 运行全部神经网络图的方式，完成了 2、4、6、8 秒音频的真实预填充、自回归生成和 EOS 终止。四组输出与原始 PyTorch CPU FP32 标准路径逐 token 完全一致，故 NPU 相对标准路径的 CER/WER 均为 **0% / 0%**。8 秒组还实际触发并验证了 `128 → 256` KV bucket 迁移，迁移后继续生成至 EOS，序列仍完全一致。

这里的“端到端”指：标准音频特征（mel）输入 → RKNN encoder → RKNN audio tower/projector → RKNN prefill → RKNN decoder 自回归至 EOS。音频文件读取、重采样/特征生成和 tokenizer 文本解码尚未包进同一个板端命令，故不把它们计入下述 neural-inference 时延。

## 已落地的动态窗口机制

原模型的 Qwen2 decoder 用 `DynamicCache`，最大 position 为 32768、没有 sliding window；本地 prefill ONNX 本身是固定长度且没有 `past_key_values` 输入。因此，RKNN 静态 shape 的部署实现以 bucket 复刻该语义：从最小可用的 `Smax=128` 开始；当写入位置达到上限时，CPU 将 8 层 K/V 的**有效区**复制到更大 buffer，切换同一层对应的 `block_256.rknn` 或 `block_512.rknn`，并从同一 token、同一 RoPE position 继续计算。`128→256` 已由自然 ASR EOS 样本触发；`256→512` 已由下文的 FP32 对照压力路径触发，尚未由自然 ASR EOS 样本触发。

跨 token 的 cache 采用 `[heads, Smax, head_dim] = [8, Smax, 64]` 的 rank-3 外部布局。CPU 只做 buffer 管理、KV 原地写入、RoPE/mask 准备、argmax 和 embedding lookup；每层新的 K/V projection、attention/FFN block、最终 RMSNorm/LM-head shard 都经 RKNN 在 NPU 上执行。没有图内 `DynamicCache`、Concat、ScatterND 或隐式 runtime cache state。

## 精度与功能验收

| 截取音频 | prefill 序列长度 | audio placeholder / adapter bucket | 生成 token 数（含首 token 与 EOS） | EOS | KV bucket | 与 CPU FP32 token | NPU vs CPU FP32 CER / WER |
|---:|---:|---:|---:|---|---|---|---|
| 2 s | 35 | 25 / T25 | 7 | 151645 | 128 | 7/7，一致 | 0% / 0% |
| 4 s | 60 | 50 / T50 | 19 | 151645 | 128 | 19/19，一致 | 0% / 0% |
| 6 s | 85 | 75 / T75 | 23 | 151645 | 128 | 23/23，一致 | 0% / 0% |
| 8 s | 110 | 100 / T100 | 23 | 151645 | 128→256 | 23/23，一致 | 0% / 0% |

`CER/WER` 使用小写、去标点、压缩空白的英文规范化；CER 不计空格。由于每条 token id 序列完全相同，文本解码也必然完全相同。标准路径中的 2/4/6 秒文本依次为：

| 时长 | 标准 FP32 文本（与 RKNN 相同） |
|---:|---|
| 2 s | `Quilter is apostle.` |
| 4 s | `Mr. Quilter is the apostle of the middle classes, and we are glad.` |
| 6 s / 8 s | `Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.` |

### 短音频问题及修复验证

首版短音频 runtime 将固定 `T100` adapter 的前 N 个 embedding 截取给 prompt。这与原模型不同：原模型会先运行 tower，再以**实际 placeholder 数**做 adaptive pooling。该错误使 2 秒组 embedding 相对 L2 达 156.97%，首 token 从 FP32 的 2183 错变为 12275；它是开发层面的 adapter pooling 形状错误，不是 decoder KV-cache 或量化精度问题。

现已将 adapter 导出为 T25、T50、T75、T100 四个静态 RKNN 图，runtime 按实际 placeholder 数选择，且强制要求 adapter 输出 token 数恰好等于 placeholder 数。板端探针的修复后结果如下。

| 时长 | adapter embedding 相对 L2（对 FP32 注入 embedding） | RKNN prefill logits 相对 L2（理想 embedding 对 FP32） | NPU frontend 相对理想 prefill logits L2 | 首 token FP32 / RKNN |
|---:|---:|---:|---:|---|
| 2 s | 0.7531% | 0.5552% | 0.5581% | 2183 / 2183 |
| 4 s | 1.1082% | 1.4813% | 0.8022% | 12275 / 12275 |
| 6 s | 0.8460% | 0.4250% | 0.6936% | 12275 / 12275 |

这些 L2 是数值健康度的辅助指标，不单独定义 ASR 是否可接受；本验收的硬判据是 greedy 自回归过程中的完整 token、EOS 与文本误差。四条均通过。

## RK3576 板端性能（稳态神经网络路径）

“稳态 E2E”是 encoder + adapter + prefill + decoder NPU 调用 + CPU KV 原地写入 + 已发生的 bucket copy；它排除了模型加载、RKNN runtime 初始化和释放。`process wall` 是整段 Python 进程壁钟时间，包含这些一次性成本，因此不能与稳态时延混用。

| 音频 | Encoder | Adapter | Prefill | Decoder NPU 总计 | CPU KV 写入 | bucket 迁移拷贝 | 稳态 E2E | RTF | process wall | RSS（结束时） |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 s | 408.55 ms | 58.06 ms | 103.95 ms | 659.53 ms | 1.40 ms | — | **1231.49 ms** | **0.616** | 10179.55 ms | 1113.15 MiB |
| 4 s | 389.64 ms | 45.97 ms | 84.42 ms | 2844.99 ms | 6.84 ms | — | **3371.87 ms** | **0.843** | 13259.35 ms | 1114.12 MiB |
| 6 s | 413.23 ms | 62.38 ms | 150.35 ms | 3482.55 ms | 8.35 ms | — | **4116.86 ms** | **0.686** | 13771.63 ms | 1116.10 MiB |
| 8 s | 405.26 ms | 58.87 ms | 165.19 ms | 3690.71 ms | 7.81 ms | **16.46 ms** (`128→256`) | **4344.29 ms** | **0.543** | 15017.83 ms | 1232.23 MiB |

8 秒组的 16.46 ms 是 CPU 对 16 个 K/V tensor 的有效内容重新分配/复制与 bucket 切换成本；不是 NPU 内部自动 cache 更新。迁移后 RSS 比未跨界的 6 秒组高约 116 MiB，符合同时驻留 S256 block 图及更大 K/V host buffer 的预期。

解码 NPU 调用内含 Host↔NPU input/output buffer 提交、NPU 执行和同步等待，当前 RKNN Lite `inference()` API 没有为这三项提供可用的逐项计时。因此本报告不伪造“搬运 xx ms”：上述 `decoder NPU 总计`是三者的合计；CPU KV 写入和 CPU bucket copy 则已独立计时。后续若要拆出 DMA/H2D/D2H，需切换至 RKNN zero-copy / memory binding API 并对 buffer 生命周期单独埋点。

## 产物与可复现入口

核心 runtime 是 [board_run_end_to_end_dynamic_buckets.py](C:/Users/Administrator/Documents/ChatGPT/asr/board_run_end_to_end_dynamic_buckets.py)，它新增 `--adapter-rknn` 并对 adapter token 数与 placeholder 数做硬校验。`export_audio_adapter_bucket.py` 已修正为按 `--tokens` 命名 ONNX，以生成 T25/T50/T75/T100 图。

原始板端结果与 CPU 标准参考：

- [2 秒 RKNN 结果](C:/Users/Administrator/Documents/ChatGPT/asr/dynamic_length_s2.json)、[2 秒 FP32 参考](C:/Users/Administrator/Documents/ChatGPT/asr/standard_length_s2.json)
- [4 秒 RKNN 结果](C:/Users/Administrator/Documents/ChatGPT/asr/dynamic_length_s4.json)、[4 秒 FP32 参考](C:/Users/Administrator/Documents/ChatGPT/asr/standard_length_s4.json)
- [6 秒 RKNN 结果](C:/Users/Administrator/Documents/ChatGPT/asr/dynamic_length_s6.json)、[6 秒 FP32 参考](C:/Users/Administrator/Documents/ChatGPT/asr/standard_length_s6.json)
- [8 秒动态跨 bucket 结果](C:/Users/Administrator/Documents/ChatGPT/asr/dynamic_length_s8.json)
- [FP32 参考生成工具](C:/Users/Administrator/Documents/ChatGPT/asr/generate_length_standard_reference.py)

## 二级迁移压力验收：256→512

在上述自然 ASR 验收之外，增加了一组**缓存调度压力测试**，目的不是评价 EOS 后文本的语义，而是让同一条真实 6 秒音频的 NPU/FP32 decoder 在首次 EOS 后按 greedy 规则继续运行，以构造超过 256 的真实 KV context。两端均执行相同的“忽略 EOS、仍将当前 argmax token 回灌”规则；这不是正常用户可见的 ASR 解码模式。

| 项目 | 结果 |
|---|---|
| 起始 prefill 长度 | 85 |
| 后续 decoder step | 180 |
| 总 token（含 prefill 首 token） | 181 |
| 实际迁移 | `128→256`（11.21 ms）；`256→512`（34.70 ms） |
| 最终 bucket | 512 |
| NPU 对 FP32 token | **181/181 完全一致** |
| 首个不一致 token | 无 |
| NPU decoder 合计 | 32611.01 ms |
| CPU KV 原地写入合计 | 61.94 ms |
| 稳态全链路合计 | 33321.24 ms |
| 结束 RSS | 1379.02 MiB |

这证明 S512 RKNN decoder block 可在迁移后读取由 CPU 复制的 rank-3 K/V buffer，并维持与 FP32 相同的贪心轨迹。其结果是**二级 cache 调度的功能和数值验收通过**；但由于该轨迹被明确设计为 EOS 后继续回灌，不能将它误解为自然语音的 WER/CER 验收。自然 ASR 部分的质量结论仍以 2/4/6/8 秒四条真实 EOS 样本的 0% NPU-vs-FP32 CER/WER 为准。

对应的可复现产物：

- [256→512 板端压力结果](</C:/Users/Administrator/Documents/ChatGPT/asr/dynamic_stress_s6_256_to_512.json>)
- [对应 FP32 强制延长参考](</C:/Users/Administrator/Documents/ChatGPT/asr/forced_decoder_reference_s6_180.json>)
- [FP32 强制延长参考生成器](</C:/Users/Administrator/Documents/ChatGPT/asr/generate_forced_decoder_reference.py>)
