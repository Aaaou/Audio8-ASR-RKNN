# Audio8-ASR：FP16 NPU 前级阶段验收

**结论：通过（限定为固定 8 秒桶和当前两条完整带标注样本）。**

本验收冻结以下实现，不包含 Qwen2 KV-cache、decode-step 或低比特量化：

```text
FP16 RKNN Audio Encoder [128,800] -> [104,1024]
  + FP16 RKNN Audio MLP Tower / pooling / Projector [104,1024] -> [100,512]
  + Qwen2 固定 S=110、无缓存 prefill 的独立 NPU 兼容验证
```

## 验收判据与结果

| 项目 | 判据 | 实测 | 判定 |
|---|---|---|---|
| Encoder RKNN placement | 模型计算层全部为 NPU；CPU 仅输入/输出边界 | 31 次图执行，每次 258 NPU 计算层；无计算 CPU fallback | 通过 |
| Adapter/projector 数值 | `normalized L2 < 0.5%`，cosine 接近 1 | 8 个校准输入：0.126%–0.147%；cosine 约 0.999999 | 通过 |
| Qwen2 prefill 独立图 | 首 token argmax 与 FP32 一致 | `S=110`：logits L2 0.3201%，argmax `12275` 一致 | 通过 |
| NPU 前级累积 embedding | 小于 2%，并检查下游首 token | 样本 0：0.8472%；样本 1：1.2329% | 通过 |
| CPU Qwen2 对前级误差的敏感性 | 首 token argmax 一致 | logits L2：0.6100% / 0.6239%；argmax 均一致 | 通过 |
| 最终转写 | 与 CPU 原始模型文本逐条一致 | 两条均逐字一致；聚合 WER 3.5714%，CER 3.1746%，与 CPU baseline 相同 | 通过 |

## 覆盖范围与限制

用于最终文本比较的两条语音均不超过 8 秒，因此没有因截断破坏 reference：

```text
1272-128104-0000: 5.855 s
1272-128104-0001: 4.815 s
```

当前 RKNN 图是静态 `[128,800]` 音频特征桶。超过 8 秒的语音需要先实现分段/多桶策略，不能把当前两条短音频结果外推为任意时长的完整 ASR 验收。

Qwen2 的 NPU prefill 图明确为 `use_cache=False`。因此它的通过仅说明完整 Qwen2 计算算子可在 NPU 上执行且首 token 保持一致，**不构成 KV-cache 自回归 decoder 已验收的证据**。

## 未开始的下一阶段

在收到下一步指示前，不修改当前已冻结路径。后续 KV-cache 阶段应独立验收：

```text
prefill: logits + 8 layers K/V cache
decode-step: previous token + historical K/V -> next logits + appended K/V
positions: 0, 1, 2, 3, 16, EOS
checks: logits, argmax, cache shape/layout/length, continuous text, CER/WER, NPU placement
```

## 可复核结果

- `audio_adapter_board_fp16.json`
- `qwen2_prefill_board_fp16.json`
- `validate_npu_frontend_cpu_qwen_s2.json`
- `end_to_end_rknn_frontend_fp16_s2.json`
- `fp16_rknn_fallback_verification.zh-CN.md`
