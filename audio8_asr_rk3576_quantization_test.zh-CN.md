# Audio8-ASR RK3576 W8A8 / W4A16 部署测试

测试日期：2026-09-13  
模型：`Edge0/Audio8-ASR-0.1B`，revision `8487da63d581fa4fc9b5c60444cb57c3a523d7aa`  
目标：Firefly ROC-RK3576，Linux 6.1.84，RKNN Runtime/Lite2 2.3.2，NPU driver 0.9.8。

## 本轮结论

本轮完成了 Audio8-ASR 音频编码器的固定 8 秒桶导出、W8A8/W4A16 构建、部署和
RK3576 实板测试。**W8A8 与 W4A16 均不可进入 ASR 端到端部署。** 两份低比特
模型都能构建、加载、初始化与执行，且输出有限值，但相对 CPU FP32 参考的误差远高于
可接受门槛。

FP16 RKNN 图可作为当前可运行的 NPU encoder 基线；但即使 FP16 的最大 normalized
L2 为 1.68%，接入 decoder 前仍需要完整 ASR 文本回归确认。当前不得宣称低比特 ASR
部署完成。

FP16 RKNN Audio Encoder 已完成单进程严格 fallback 核验：31 次实际执行（1 次预热 + 30 次
计时）中的 258 个模型计算层均由 RKNPU 执行；CPU 仅处理每次的输入/输出边界和 Runtime
辅助工作。详见 [FP16 RKNN CPU fallback 严格核验](fp16_rknn_fallback_verification.zh-CN.md)。

| 图 | `.rknn` 大小 | 平均板端推理 | 峰值进程 RSS | 最低 cosine | 最大 normalized L2 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| FP16 | 372 MiB | 408.29 ms | 889.1 MiB | 0.999860 | 1.680% | 可用作 encoder 图基线，不等同端到端验收 |
| W8A8 | 201 MiB | 307.07 ms | 524.3 MiB | 0.872592 | 50.092% | 拒绝 |
| W4A16 | 111 MiB | 358.16 ms | 392.2 MiB | 0.849849 | 57.572% | 拒绝 |

W8A8 虽比 FP16 快约 25%，W4A16 虽将磁盘模型进一步降至 111 MiB，但两者的数值
误差足以改变下游 audio embedding，不能用于转写。W4A16 在这张图上还比 W8A8 慢。

RSS 采样结果为单独 Python 推理进程的 `ru_maxrss`：FP16 在 `load/init/peak` 后依次为
`410.9/867.0/889.1 MiB`，W8A8 为 `239.9/500.5/524.3 MiB`，W4A16 为
`150.7/267.8/392.2 MiB`。它不含内核 CMA、DMA/IOMMU 映射和 NPU 专用内存，不能当作
设备总内存占用。Runtime verbose 日志确认 FP16 图 target 为 `RKNPU f2`，并分配了
`387,419,776 bytes` DMA weight 与 `80,691,264 bytes` internal buffer；日志同时明确
该图没有 GPU op，因此这些测试是 RK3576 RKNN Runtime/NPU 路径，而不是 ONNX Runtime CPU。

## FP16 CPU 与 FP16 RKNN 对照

为避免把主机侧 CPU 数据误认为板端结果，2026-09-13 在同一 RK3576 板上补测了固定
`[128, 800] -> [104, 1024]` encoder 图的 CPU FP16 路径。CPU 使用板端 ONNX Runtime
1.23.2 的 `CPUExecutionProvider`，输入、权重与输出均为 FP16；`intra_op_num_threads=4`、
`inter_op_num_threads=1`，每个样本预热一次后测 2 次。RKNN NPU 数据为同一批 8 个真实
mel 输入、每个样本预热一次后测 3 次。

| 路径 | 后端 | 平均推理 | 峰值进程 RSS | 最大 normalized L2（相对原始 CPU FP32 参考） |
|---|---|---:|---:|---:|
| FP16 CPU | ONNX Runtime CPUExecutionProvider | 2770.83 ms | 1767.1 MiB | 0.0935% |
| FP16 RKNN | RKNNLite / RKNPU f2 | 408.29 ms | 889.1 MiB | 1.6800% |

在此固定 encoder 基准上，FP16 RKNN NPU 平均时延为 CPU FP16 的 **6.79x** 更快，进程可见
峰值 RSS 约低 **878.0 MiB**。这不是完整 ASR 的端到端加速比：mel 提取、audio embedding
注入、Qwen2 decoder、KV cache 与生成循环都不在此对比范围内。

CPU FP16 的最大 L2 远低于 RKNN FP16，说明 CPU FP16 图本身数值稳定；RKNN FP16 的
1.18%--1.68% 差异来自 RKNN/NPU 图的 FP16 执行路径，仍必须以端到端文本 CER/WER 回归
决定是否可上线。两边的 RSS 均是进程 `ru_maxrss`，不含内核侧 CMA、DMA/IOMMU 与 NPU
专用内存，不能用于完整设备内存预算。

## Ryzen 5 5600 本机 FP16 CPU 基准

在本机 AMD Ryzen 5 5600（6 物理核 / 12 逻辑线程，32 GiB 内存）的 WSL2 Ubuntu 环境，
使用 ONNX Runtime 1.26.0 `CPUExecutionProvider` 对同一 FP16 ONNX encoder 图和相同 8
个 8 秒输入进行测试。每个样本预热一次后重复 5 次；下表为 8 个样本的平均值。该结果反映
本机 CPU 的计算性能，不含 Windows 原生运行时、音频预处理、decoder 或文本生成。

| ORT intra-op 线程数 | 平均推理 | 平均最小推理 | 峰值进程 RSS | 最大 normalized L2 |
|---:|---:|---:|---:|---:|
| 1 | 693.68 ms | 677.54 ms | 1396.3 MiB | 0.1006% |
| 6（物理核） | **254.87 ms** | **242.70 ms** | 1398.2 MiB | 0.1006% |
| 12（含 SMT） | 360.26 ms | 322.36 ms | 1402.3 MiB | 0.1006% |

6 个物理核是本机最佳配置：相对 RK3576 FP16 RKNN encoder 的约 408--410 ms，快约 **1.6x**；
相对 RK3576 板端 CPU FP16 的 2770.83 ms，快约 **10.9x**。12 线程较 6 线程慢约 41%，部署
时应设置 ONNX Runtime `intra_op_num_threads=6`，而非机械地使用所有逻辑线程。

## 已部署的板端材料

部署目录：`/root/audio8-asr/encoder_f800_v2/`

```text
audio_encoder_f800_fp16.rknn
audio_encoder_f800_w8a8.rknn
audio_encoder_f800_w4a16.rknn
calibration/sample_00..07_input.npy
calibration/sample_00..07_reference.npy
board_test_audio_encoder.py
board_fp16.json
board_w8a8.json
board_w4a16.json
board_cpu_fp16_ort.json
```

旧目录 `/root/audio8-asr/encoder_f800/` 保留首轮单一音频校准的产物，仅用于复盘，
不作为任何结果的依据。

## 图与校准集

### 静态化策略

原模型的 `Qwen3ASRAudioEncoder.forward` 含有按真实长度执行的 `split`、
`pad_sequence`、布尔 packing、Python 循环与动态 `cu_seqlens`。第一次直接导出因
`aten::pad_sequence` 无 ONNX exporter 支持而失败。

当前图固定为 800 个 mel frame（8 秒，128-bin mel）：

```text
[128, 800] mel
-> 8 x 固定 100-frame chunk
-> 3 x stride-2 Conv2d
-> 每个 chunk 13 token
-> 104 token 的 18 层 audio Transformer
-> [104, 1024] audio_hidden
```

静态图在导出前与上游动态 encoder 的最大绝对误差低于 `1e-5`。ONNX Runtime 对 8 组
CPU 参考输出的 normalized L2 为 `3.33e-6--5.80e-6`，说明静态化没有引入实际误差。

图仍包含 18 组注意力链路，包括 18 个 `ScatterND`、18 个 `Softmax`、37 个 `MatMul`
和 110 个 `Gemm`。RKNN FP16 实板结果证明这些固定形状路径在当前 Runtime/driver
组合上可以运行。

### 校准样本

校准集使用 8 段不同的真实语音波形，重采样至 16 kHz 后生成固定 8 秒 mel。每个 `.npy`
的 SHA-256 已确认不同。首轮仅用 `mia.wav` 时，因为重采样后的音频小于 8 秒，所有
输入被零填充成同一数组；该问题已发现并废弃，未用于本报告的 W8/W4 结论。

## 板端验证方法

每个模型均在同一板、同一 Runtime 版本下执行：

1. `load_rknn()`；
2. `init_runtime()`；
3. 每个样本预热一次；
4. 每个样本连续推理 3 次，记录平均/最小延迟；
5. 将最后一次输出与 CPU FP32 参考比较 `finite`、cosine、normalized L2、MAE 和
   最大绝对误差。

Lite2 出现的“Query dynamic range failed”只针对静态 shape 图的动态范围查询，不影响
实际加载、推理或此处的数值结果。

原始结果：

- [FP16 board report](C:/Users/Administrator/Documents/ChatGPT/asr/board_fp16.json)
- [W8A8 board report](C:/Users/Administrator/Documents/ChatGPT/asr/board_w8a8.json)
- [W4A16 board report](C:/Users/Administrator/Documents/ChatGPT/asr/board_w4a16.json)
- [FP16 CPU board report](C:/Users/Administrator/Documents/ChatGPT/asr/board_cpu_fp16_ort.json)
- [Ryzen 5 5600 FP16 CPU, 1 thread](C:/Users/Administrator/Documents/ChatGPT/asr/host_ryzen5600_fp16_ort_t1.json)
- [Ryzen 5 5600 FP16 CPU, 6 threads](C:/Users/Administrator/Documents/ChatGPT/asr/host_ryzen5600_fp16_ort_t6.json)
- [Ryzen 5 5600 FP16 CPU, 12 threads](C:/Users/Administrator/Documents/ChatGPT/asr/host_ryzen5600_fp16_ort_t12.json)

## 为什么当前 W8A8 / W4A16 被拒绝

生产门设置为：所有输出 finite、cosine >= 0.99、normalized L2 <= 1%，并在接入
decoder 后验证贪心文本 token 一致。W8A8 的 L2 为 45.65%--50.09%，W4A16 为
42.01%--57.57%，均比门槛高两个数量级。

FP16 相对参考的 1.18%--1.68% 误差主要是 RKNN 的 FP16 内部执行精度，不是异常的
布局或全零输出；相反，W8/W4 的额外巨大误差表明针对完整 attention encoder 的全图
激活/权重量化在此工具链上不可接受。不能因为它们模型更小或执行更快而继续接入。

## 后续技术路线

1. 保留 `audio_encoder_f800_fp16.rknn` 作为 NPU encoder 基线，开发 CPU audio
   embedding 注入与 CPU Qwen2 decoder 的端到端混合运行时，并以真实音频验证文本。
2. 不再尝试同一整图的单阶段全量 W8A8/W4A16 参数微调。当前误差规模表明需要改变
   量化边界，而不是仅增加几条校准样本。
3. 若继续量化，先以 RKNN two-stage hybrid quantization 保留 attention score、
   Softmax、LayerNorm、Q/K/V 与输出投影为 FP16，仅量化 FFN 的线性层；每个候选均
   需重复本报告的 8 样本实板门槛。
4. 如果 hybrid 仍不能在 <=1% L2 下明显快于 408 ms，则停止 encoder 低比特路线，
   以 FP16 encoder 或 CPU ONNX Runtime 作为部署实现。
5. decoder 的 KV cache、采样与 token loop 仍留在 CPU，直到独立的 prefill/decode
   图通过 position 0/1/3/16、连续 token 和最终 CER/WER 测试。

## 可重复脚本

- [固定桶导出与真实校准集生成](C:/Users/Administrator/Documents/ChatGPT/asr/export_audio_encoder_bucket.py)
- [RKNN W8A8/W4A16/FP16 构建](C:/Users/Administrator/Documents/ChatGPT/asr/build_rknn_quantized.py)
- [ONNX CPU 回归](C:/Users/Administrator/Documents/ChatGPT/asr/validate_onnx_bucket.py)
- [RK3576 板端测试](C:/Users/Administrator/Documents/ChatGPT/asr/board_test_audio_encoder.py)
- [RK3576 CPU FP16 对照测试](C:/Users/Administrator/Documents/ChatGPT/asr/board_benchmark_onnx_cpu.py)
- [FP16 RKNN CPU fallback 严格核验](C:/Users/Administrator/Documents/ChatGPT/asr/fp16_rknn_fallback_verification.zh-CN.md)
