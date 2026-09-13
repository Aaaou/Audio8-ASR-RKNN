# Qwen2 KV-cache decode-step：补测与优化状态

## 补测结果

当前 `S=110` 图的板端调用：

| 项目 | FP32 输入 | FP16 输入 |
|---|---:|---:|
| RKNN inference mean | 97.68 ms | 96.55 ms |
| logits normalized L2 | 113.16% | 113.16% |
| next token | 3711 | 3711 |
| reference next token | 13 | 13 |

将输入 cache/token 从 FP32 改为 FP16 后结果完全不变，因此错误不是输入边界精度转换造成的。

## 时间与内存

FP32 输入下：

```text
RKNN inference total: 97.68 ms
CPU cache concatenate: 3.63 ms
CPU cache concatenate P95: 5.76 ms
RSS before RKNN init: 43.95 MiB
RSS after RKNN init: 453.96 MiB
RSS after 20 runs: 481.45 MiB
CPU->NPU historical K/V: 3,604,480 bytes (~3.44 MiB)
NPU->CPU logits: 607,744 bytes (~0.58 MiB)
NPU->CPU K/V delta: 32,768 bytes (32 KiB)
```

普通 `RKNNLite.inference()` 没有把 DMA、layout conversion、driver submit、同步和 NPU operator compute 分开暴露；因此 97.68 ms 目前只能报告为混合总调用时间，不能冒充纯搬运时间。CPU concat 是单独可测的 3.63 ms。

## 数值判定

本项目 KV-cache 的严格门槛为：

```text
decode logits normalized L2 <= 0.5%：通过候选
0.5%~1%：警告，必须扩大 token/样本验证
>1% 或 top-1 token 改变：不通过
```

当前结果：

```text
logits L2 = 113.16%
K/V delta 最大 L2 = 143.56%
next token 13 -> 3711
```

这是明确的图语义失败，不是轻微 FP16 损失。

## 当前优化结论

CPU cache 拼接不是当前瓶颈，也不是错误来源。错误出现在 `ONNX -> RKNN compile/runtime` 的 cache attention 路径，重点怀疑 `[B,H,S,D]` 外部 cache 的 layout/transpose/Gather 融合。尚未构建 `S=111`，因为在 `S=110` 单步未通过前继续扩展只会放大错误。

下一修复候选是让 RKNN 外部 cache 使用 `[B,S,H,D]`，图内显式 transpose 为 Qwen2 的 `[B,H,S,D]`，并重新做 ONNX、RKNN 单步对照；若仍失败，再拆成逐层 attention 图定位首个偏差层。
