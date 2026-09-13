# Audio8-ASR：完整 RKNN 自回归端到端验收（样本 1）

## 结论

已完成两条音频从音频特征到真实 EOS 的完整 RKNN 推理闭环。两条的 greedy token 序列均与标准 CPU FP32 / ONNX 语义 **逐 token 完全一致**，最终文本完全一致，因此 RKNN 相对标准路径的聚合 CER/WER 均为 **0%**。

## 实际运行路径

```text
输入音频的 128×800 mel 特征
→ Audio Encoder FP16 RKNN / NPU
→ Audio MLP tower + projector FP16 RKNN / NPU
→ 将 100 个音频 embedding 放入 prompt embedding
→ Qwen2 FP16 RKNN prefill（输出 logits 和 8 层初始 K/V）
→ 首 token
→ 自回归循环至 EOS
   CPU: embedding lookup、RoPE/mask、K/V buffer 原地写、argmax
   NPU: 8 K/V projections + 8 attention/FFN blocks + 8 LM-head shards
→ EOS=151645
→ tokenizer decode
```

没有使用 PyTorch decoder、预先保存的 decoder cache、预先回放的 hidden state 或 CPU decoder 计算。CPU 持有 K/V 是部署设计的一部分；NPU 不具有通过普通 `inference()` 跨调用自动持久化 cache 的对象。

## 文本和 token 对照

| 样本 | token 数（含 EOS） | token / EOS 对比 | 最终文本 | RKNN 相对基线 CER / WER |
|---|---:|---|---|---|
| `1272-128104-0000` | 23 | 23/23 一致；EOS=151645 | `Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.` | 0% / 0% |
| `1272-128104-0001` | 14 | 14/14 一致；EOS=151645 | `Nor is Mister Quilter's manner less interesting than his matter.` | 0% / 0% |
| 合计 | 37 | 37/37 一致 | 两条均一致 | **0% / 0%** |

相对 LibriSpeech reference 的聚合 CER/WER 为 **3.175% / 3.571%**，与标准 CPU FP32 的既有基线完全相同。

第二行的 CER/WER 是模型自身把 `MISTER` 写为 `Mr.` 所产生的基线误差；由于两条路径文本完全相同，不能归因于 RKNN。

## 性能与内存

板端为 RK3576、RKNN Runtime 2.3.2、RKNPU driver 0.9.8。缓存 bucket 为 Smax=256；audio prefill 占 110 个位置，生成至 EOS 共 23 token，未触及容量上限。

| 阶段 | 时间 |
|---|---:|
| Audio Encoder NPU | 400.70 ms |
| Audio adapter/projector NPU | 57.17 ms |
| Qwen2 prefill NPU（含 16 个 K/V 输出） | 131.51 ms |
| Decoder NPU 平均 | 186.31 ms / token |
| CPU K/V write 平均 | 0.385 ms / token |
| 常驻模型后纯计算估算 | 约 4.69 s / 此样本 |
| 本次进程总墙钟 | 13.35 s |
| RSS（运行前 / 运行后） | 35.24 / 1131.23 MiB |

13.35 秒包含 RKNN 图初始化、首次加载 297 MiB token embedding 表和 tokenizer/调度初始化，**不应**和常驻服务中的纯推理延迟混为一谈。CPU KV 写入只约为 decode 总时间的 0.2%，不是性能瓶颈；当前瓶颈是每 token 需要执行 8 个 layer block 加 8 个 LM-head shard。

## Placement

所有神经网络计算在 RKNN NPU：encoder、adapter、prefill、K/V projection、attention、FFN、final norm 和 LM head shards。CPU 只作 embedding lookup、RoPE/mask 生成、K/V 内存写入、argmax 与文本解码。静态 RKNN 的 dynamic-range 查询警告不代表 CPU fallback。

## 当前边界

本轮已满足“像标准 ONNX 一样，从输入经自回归生成直到 EOS 并得到完整文本”的功能要求，两个 8 秒样本均已完成。当前仍应继续扩大到长音频、多语言、热词及超过 256 token 的 bucket，以获得生产范围的置信度。
