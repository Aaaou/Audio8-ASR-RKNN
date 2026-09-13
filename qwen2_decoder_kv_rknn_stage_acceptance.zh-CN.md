# Qwen2 decoder：RK3576 KV-cache NPU 调度阶段验收

## 结论

**功能链路通过；严格数值门槛为“警告通过”，尚不应称为全量语料精度上线通过。**

已完成 8 个 decoder layer、K/V 投影、attention、FFN、final RMSNorm 和全量 LM head 的 RKNN 部署。LM head 因词表为 151,936 而分成 8 个 NPU shard；没有把其矩阵计算退回 CPU。

此前首个 decode 图失败并不是模型本身不支持 KV-cache，而是 `[B,H,S,D]` 四维 cache input 在 RKNN Lite 的布局边界被错误解释，导致历史 K/V 在进入 attention 前即失真。当前实现将外部缓存改成 `[H,S,D]`，使用固定 128 槽 CPU buffer 和显式 attention mask；图内没有 DynamicCache、Concat、Scatter 或跨调用隐式 state。

## 已验证调度契约

```text
CPU:  固定 K/V buffer、当前位置、in-place K/V 写入、token argmax、embedding lookup、RoPE 输入准备
NPU:  每 token 的 8×K/V projection、8×attention+FFN block、8×LM-head shard
```

缓存本体在 CPU，格式为 8 layers × K/V × `[8,128,64]` FP32，容量 **4,194,304 bytes / 4.00 MiB**。每步仅在 CPU 向对应位置写入每层的新 K/V；不是 CPU 重新拼接整段历史缓存，也不是 NPU 图内更新缓存。

## 严格板端连续调度结果

环境：RK3576、RKNN Runtime 2.3.2、RKNPU driver 0.9.8、FP16 RKNN、8 秒英文样本。预填充长度为 110；固定 cache 上限为 128，因此本轮覆盖接下来的 **18 个连续 decode token**。每一步的下一个输入 embedding 都由前一步 NPU logits 的实际 argmax 查表获得，而非回放隐藏状态。

NPU 与 CPU FP32 greedy reference 的 18-token 序列逐项一致：

```text
12275 → 13 → 3406 → 2044 → 374 → 279 → 38471 → 273 → 315 → 279
      → 6149 → 6846 → 11 → 323 → 582 → 525 → 15713 → 311 → 10565
```

| 项目 | 结果 | 验收解释 |
|---|---:|---|
| token 序列一致 | 18 / 18 | 通过 |
| 最大 K delta normalized L2 | 0.5017% | 0.5% 目标线略超，处于 0.5–1% 警告带 |
| 最大 V delta normalized L2 | 0.7877% | 同上，未超过 1% 禁止部署线 |
| 最大层输出 normalized L2 | 0.5596% | 同上 |
| 首步完整 logits L2 | 0.3954% | 通过 0.5% 目标线，top-1=13 一致 |
| CPU cache write 平均 | 0.046 ms / token | 可接受；不是延迟瓶颈 |
| NPU 总计算平均 | 154.30 ms / token | 当前性能瓶颈；包括 8 个 block 和 8 个 head shard |
| 进程 RSS（初始化后 / 测后） | 458.69 / 471.54 MiB | 不含内核 CMA/DMA |

单次 NPU 调用平均：K/V projection 2.64 ms、decoder block 11.47 ms、LM-head shard 5.18 ms。后两者各有 8 次调用，因此约为 21.1 + 91.8 + 41.4 = 154.3 ms/token。CPU 的 cache 写入只占约 0.24%。

## NPU placement 核验

RKNN 编译表中，KV projection 和 block 内的 RMSNorm、Q/K/V/O 线性层、RoPE、attention（`exSDPAttention`）、MLP 的 gate/up/down 均标为 **NPU**；head shard 的 RMSNorm 与 512×18,992 linear 也标为 **NPU**。CPU 节点仅为 RKNN 的输入/输出边界，以及上述调度职责。运行时的 “Query dynamic range failed” 是静态 shape RKNN 查询动态范围时的已知提示，不代表 fallback。

## 本阶段的验收问题

1. **数值仍在警告带而非绿色带。** 虽然 token 18/18 一致，最大 K/V/hidden L2 分别是 0.5017% / 0.7877% / 0.5596%。按预设标准（≤0.5% 绿、0.5–1% 警告、>1% 失败），本阶段只能以“功能正确、精度待扩样本确认”验收，不能据此宣称 CER/WER 已完成。
2. **128 token 是当前静态 cache 容量，不是无限长度。** 预填充为 110 时只剩 18 个生成位置；生产版本需要按最大生成长度导出更大 Smax，或按 bucket 管理 128/256/512 等规格。该行为是静态图设计约束，不是 KV 调度错误。
3. **尚未做完整音频集的 decoder-NPU CER/WER。** 本次 18 token 没有到 EOS，不能形成完整转写，故不能诚实地给出 decoder-NPU 的 CER/WER。此前已通过的是“前端 NPU + CPU decoder”两样本端到端文本一致；本阶段需要 Smax 扩展后运行到 EOS，再做多样本 CER/WER。

## 可复现产物

- `qwen2_decoder_kv_rknn_stage_acceptance.json`：板端 18-step 原始验收数据。
- `export_decoder_3d_fixed_cache.py`：8 层固定 buffer、rank-3 cache 图导出。
- `board_test_decoder_3d_fixed_schedule.py`：实际 argmax 驱动的 CPU-cache / NPU-decode 调度核验。
- `prepare_decoder_schedule_refs.py`：CPU FP32 逐层 reference 生成。

下一阶段应优先将 Smax 扩展至能跑到 EOS 的长度，并在至少两条真实音频上做完整贪心生成及 CER/WER；只有在数值和文本均保持稳定后才标记为生产验收通过。
