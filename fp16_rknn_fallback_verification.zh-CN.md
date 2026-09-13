# Audio8-ASR FP16 RKNN CPU Fallback 严格核验

测试日期：2026-09-13  
目标：Firefly ROC-RK3576，RKNN Runtime/Lite2 2.3.2，RKNPU driver 0.9.8。  
图：`audio_encoder_f800_fp16.rknn`，固定输入 `[128,800]` FLOAT16，输出 `[104,1024]`。

## 结论

Audio8-ASR 的 FP16 RKNN Audio Encoder **没有发生模型计算的 CPU fallback**。完整主图的
计算层均被 Runtime 放置在 `RKNPU f2`；CPU 只参与输入/输出边界、输入归一化和 RKNN
Runtime 调度/同步等辅助工作。

## 严格测试方法

1. 确认板端无残留 RKNN/probe 进程，RKNPU Core0/Core1 均为 0%，频率为 300 MHz；
2. 单独启动一个测试进程，加载 FP16 `.rknn`；
3. 对同一个真实 calibration mel 输入预热一次，再连续同步执行 30 次；
4. 进程内每 20 ms 读取 `/sys/kernel/debug/rknpu/load` 与 `freq`；
5. 使用 RKNN verbose layer table 检查每个图层的 `Target`；
6. 搜索 Runtime 日志中的 CPU/GPU fallback 与 CPU op 记录。

## 运行结果

| 项目 | 结果 |
|---|---:|
| Runtime target | `RKNPU f2` |
| Target platform | `rk3576` |
| 图类型 | ONNX / `static_shape` |
| 计时推理次数 | 30 次，另有 1 次 warm-up |
| 总计时墙钟 | 13.4456 s |
| 平均单次同步推理 | 448.19 ms |
| 测试进程 CPU 时间 | 2.69 s |
| 测试进程平均 CPU 利用率 | 20.01%（相对单个逻辑核） |
| RKNPU load 采样数 | 627 |
| 非零 RKNPU load 样本 | 627 / 627（100%） |
| RKNPU peak load | 87% |
| peak frequency | 300 MHz |

进程 CPU 时间仅为推理墙钟的约 20%，与 NPU 同步等待、输入输出 DMA/归一化和 Python/NumPy
管理开销一致；它不能解释约 13.45 秒的主图计算。

## 图层放置证据

日志显示 warm-up 加 30 次执行共 31 次网络层表输出：

| Runtime 图层 Target | 记录数 | 含义 |
|---|---:|---|
| NPU | 7,998 | `31 × 258`，即每次 258 个模型计算层均在 NPU。 |
| CPU | 62 | `31 × 2`，每次仅 `InputOperator:input_features` 和 `OutputOperator:audio_hidden`。 |

NPU 层覆盖 Conv/ConvAdd/ConvExGelu、`exNorm`、Q/K/V projection、`exSDPAttention`、
FFN、reshape/transpose 等 Audio Encoder 图操作。典型日志：

```text
ConvExGelu  FLOAT16  NPU
exNorm      FLOAT16  NPU
Conv         FLOAT16  NPU  ... q_proj/k_proj/v_proj
exSDPAttention FLOAT16 NPU
ConvAdd      FLOAT16  NPU  ... fc2
```

CPU 记录是明确的 I/O 边界，不是模型算子 fallback：

```text
InputOperator  FLOAT16  CPU  ... InputOperator:input_features
OutputOperator FLOAT16  CPU  ... OutputOperator:audio_hidden
```

`normalize target: CPU` 也在每次推理前出现；它对应 Runtime 的输入归一化/边界辅助步骤。它
不在 Network Layer Table 的 258 个模型计算层内，且不是 `fallback CPU` 记录。

## Fallback/GPU 检查

Runtime 日志没有出现下列任何模型计算 fallback 证据：

```text
fallback CPU
CPU fallback
runs on CPU
CPU op
```

同时 Runtime 明确记录：

```text
The RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU is not set
and without GPU op in Graphs, OpenCL will not be initialized
```

因此本图没有 GPU/OpenCL 路径，且无 CPU 模型算子 fallback。静态图触发的
`RKNN_QUERY_INPUT_DYNAMIC_RANGE` warning 只说明动态范围查询不适用于 static shape，未影响
加载、层放置或 30 次有效推理。

## 范围

本结论仅覆盖已导出的 Audio Encoder RKNN 图。WAV/PCM 解码、重采样、mel 特征、Audio MLP
Tower、audio projector、Qwen2 prefill、Qwen2 KV-cache decode 与 detokenizer 仍不在该
RKNN 图内，当前由 CPU 执行。
