# RK3576 / RKNN KV-cache 策略调研

## 结论

RK3576 NPU 可以执行带有 cache 输入、Concat、Gather/Slice、Attention 的静态图；因此不能说硬件“完全不支持 KV cache”。但 RKNN Lite 普通 `inference()` 是无状态函数接口，没有一个通用的、跨调用自动持久化的 KV-cache 对象。所谓“由 NPU 自己更新 cache”必须在图中显式实现写回（Concat、ScatterND、预分配 buffer + position 等），并不等同于 runtime 自动保存状态。

当前项目第一版用了 Transformers `DynamicCache.update()` 导出路径。该路径把 cache 更新/Concat trace 进 ONNX，再由 RKNN 重写 layout。它在 ONNX Runtime 上正确，但 RK3576 上第 0 层的 projection 正常、Concat 后即出现约 140% L2 偏差，说明当前问题是编译后 cache layout/reorder/Gather/Concat 语义，而非 RK3576 缺少 attention 算力。

本地 TTS RK3576 研究也记录了同类事实：完整 AR FP16 图曾在实板 `rs-gather` submit 失败；问题收敛到 compiler 为 attention / ScatterND 生成的固定重排 Gather 路径。另一个 Fast AR 图在 position 1 分叉；无状态 FFN 子图则稳定。这些记录支持“状态写回/重排路径是风险中心”的判断。

## 三种实现路线

### 1. NPU 图内动态 Concat/ScatterND

```text
NPU 读取旧 cache → NPU 计算新 K/V → NPU Concat/Scatter 写回 → Attention
```

优点：理论上减少 CPU 拼接，数据路径短。

风险：当前已经失败；容易触发 RKNN 的 `-rs-gather`、layout 重排、ScatterND 和静态轴问题。即使 compiler 能生成 `.rknn`，也必须逐位置验证，不能以首轮能运行作为通过标准。

### 2. CPU 管理 cache，NPU 读只读历史 cache（推荐第一修复路线）

```text
CPU 保存 old K/V
NPU 计算当前 token 的 attention、FFN、logits 和 new K/V
CPU concat(old, new_delta)
```

优点：跨调用状态明确，CPU concat 只有约 3.63 ms；失败时边界清楚，可按 token 调试。缺点：每步需要搬运历史 K/V，且 decoder 图需要避免再次调用 `DynamicCache.update()` 的内部 Concat。

### 3. 固定最大 cache + NPU 位置写入

```text
预分配 [B,H,Smax,D]，用 cache_position 写入当前位置
```

优点：固定 shape，有机会实现 zero-copy/少搬运。

风险：仍依赖 Scatter/Masked assignment、position 和 layout；应在路线 2 证明数学递归正确后再尝试。

## 当前推荐顺序

```text
先修正 decoder 图，去除 DynamicCache.update() 的图内写回/Concat；
让 CPU 跨步保存并追加 K/V，NPU 图只消费明确的只读历史 cache；
先做第 0 层和 token 1 的逐中间量比较；
通过后扩展 8 层与 token 0/1/2/3/16/EOS；
最后再试固定最大 cache 的 NPU 写入。
```

## 验收门槛

KV cache 是递归状态，门槛比一次性 encoder 更严格：

```text
每层 K/V delta normalized L2 <= 0.5%：候选通过
0.5%~1%：警告，不能直接部署
>1%：不通过
decode logits normalized L2 <= 0.5%，且 top-1 token 不变
token 0/1/2/3/16/EOS 全部一致
最终 CER/WER 不出现稳定退化
```

当前图：logits L2 113.16%、K/V delta 最大 L2 143.56%、token 13→3711，属于明确失败。

## 时间解释

当前 `97.68 ms` 是普通 `rknn.inference()` 混合总时间，包含输入准备、runtime/layout、driver submit、NPU 计算、输出回收和同步；不能称为纯搬运。单独可测的 CPU concat 是 3.63 ms，历史 K/V 输入约 3.44 MiB，delta 输出 32 KiB。后续应继续使用 runtime layer table 报告 NPU operator time 和 CPU boundary time，把剩余量称为 runtime/data-boundary overhead，而不是 DMA 纯时间。
