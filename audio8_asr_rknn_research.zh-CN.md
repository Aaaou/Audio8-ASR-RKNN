# Audio8-ASR-0.1B 架构与 RKNN 部署调研

调研日期：2026-09-13  
对象：`Edge0/Audio8-ASR-0.1B`（仓库 revision
`8487da63d581fa4fc9b5c60444cb57c3a523d7aa`）  
对照：本机 `C:\Users\Administrator\Documents\ChatGPT\TTS\Audio8_TTS\rk3576`
的 Audio8-TTS 0.1B / RK3576 实测记录。

## 结论

**ASR-0.1B 不含 Mamba、Falcon-H1、SSM 或 selective scan。** 它与同厂
Audio8-TTS 0.1B 的慢速 AR 主干不是同一个架构。因此，TTS 中“显式 Mamba
conv/SSM state + 混合注意力”带来的状态接口问题不会直接出现在 ASR 模型上。

但这不等于 ASR 可直接完整转换到 RKNN NPU。其主要难点改为：

1. 音频编码器在 Python 前向中依据真实音频长度动态分块、padding、布尔索引并构造
   block-diagonal attention mask；这不适合直接导出一个泛化的动态 ONNX/RKNN 图。
2. 18 层音频 Transformer 的 attention 显式生成 `Q x K^T` 和全矩阵 mask；静态化
   后仍有长序列 `MatMul + Add + Softmax + MatMul`、`Gather/Scatter/Reshape` 等布局风险。
3. 文字解码器是 8 层 Qwen2 causal LM，生成阶段依旧有 **KV cache** 和逐 token
   递归。它没有 Mamba state，却仍会触发 TTS 已实测过的 cache 重排/Gather、浮点
   递归误差累积和 CPU<->NPU 小张量调用开销风险。
4. `vocab_size=151936`、`hidden_size=512` 的 LM head 单次输出约 151,936 logits；
   把完整词表头和采样循环纳入 NPU 对端侧内存、输出搬运和延迟都不友好。

**当前 RK3576/RKNN2.3.2 工程建议：不要以“整模型动态 ONNX -> 一个 RKNN”作为首条路径。**
先交付 CPU/ONNX Runtime 基线；NPU 尝试以固定时长、固定 batch 的音频 encoder
prefill 静态图为第一优先级。文字 decoder 应将 cache、位置/掩码构造、采样和 token
循环保留 CPU，只有在端到端数据证明收益时才逐步迁移无状态计算块。

另一个容量上的重要事实：`0.1B` 是该 checkpoint 的产品名称，不能直接作为整包
参数或内存预算。HF safetensors 元数据记录的整模型参数总数为 **323,990,528**，
全部为 BF16；仅权重原始体积就约 618 MiB（不含运行时工作区、KV cache 与 NPU
权重副本）。

## 证据：实际模型架构

模型卡 API 声明架构为 `ArkasrForConditionalGeneration`，权重为 BF16，参数总数
323,990,528（约 3.24 亿参数），使用 remote custom code，而不是标准单一
Transformers 模型：

- 模型仓库：[Edge0/Audio8-ASR-0.1B](https://huggingface.co/Edge0/Audio8-ASR-0.1B)
- 配置：[config.json](https://huggingface.co/Edge0/Audio8-ASR-0.1B/blob/main/config.json)
- 主模型代码：[modeling_arkasr.py](https://huggingface.co/Edge0/Audio8-ASR-0.1B/blob/main/modeling_arkasr.py)
- 音频编码器：[qwen3_asr_audio_model.py](https://huggingface.co/Edge0/Audio8-ASR-0.1B/blob/main/qwen3_asr_audio_model.py)

| 部分 | 结构和配置 | Mamba/循环状态 | RKNN 含义 |
|---|---|---|---|
| 特征输入 | 128-bin mel，最长 1500 frames | 无 | mel 提取可留 CPU；输入长度必须在导出前固定或分档 |
| Audio encoder 前端 | 3 个 `Conv2d(3x3, stride=2)`，通道 1 -> 480 -> 480 -> 480；随后 Linear 到 896 | 无 | Conv 适合 NPU；动态 chunk/pad/布尔索引不适合直接下沉 |
| Audio encoder 主干 | 18 个 Transformer encoder layer；`d_model=896`，14 heads，FFN `896 -> 3584 -> 896`，GELU、LayerNorm | 无 | 有 18 组全注意力；主要 prefill 计算候选 |
| Audio 输出 | LN + `896 -> 896 -> 1024` 投影 | 无 | 可与 encoder 合图，或保持为独立静态图 |
| Audio bridge | 4 个残差 MLP block，`1024 -> 4096 -> 1024`，GELU + LayerNorm；最后 `1024 -> 512` 投影 | 无 | 纯线性/归一化，较适合从大图中抽取 |
| Text decoder | `Qwen2ForCausalLM`，8 层 full attention，hidden 512，8 Q/KV heads，FFN 1408，SiLU，RMSNorm；词嵌入与 LM head 绑定 | **仅 KV cache** | 自回归单 token 调用；与 TTS Mamba 不同，但仍须处理 KV cache |
| 生成控制 | HF `GenerationMixin`；后续 step 只送最后一个 token，复用 `past_key_values` | KV cache | 采样、EOS、热词/文本后处理应留 CPU |

### 为什么可以确定没有 Mamba

1. ASR checkpoint 的顶层 `layer_types` 是 8 个 `full_attention`；没有
   `mamba_*`、`ssm_*`、`FalconH1` 或 `selective_scan` 配置字段。
2. 其 custom model 直接实例化 `Qwen2ForCausalLM` 与 `Qwen3ASRAudioEncoder`。
3. 音频层源码是常规 `q_proj/k_proj/v_proj/out_proj`、`torch.matmul`、`softmax`
   和两层 FFN；不存在状态空间递推或 scan 算子。
4. 对照的 TTS 0.1B 明确标为 Falcon-H1 `Mamba + attention` hybrid，维护
   `conv_states`、`ssm_states` 及 attention KV state。两者不能混为一谈。

## 直接影响 RKNN 的算子与图形问题

### A. ASR 特有的导出障碍：动态音频预处理写在模型 forward 内

音频 encoder 的 `forward` 会将 `feature_lens` 转为 Python 整数/列表，按
`n_window=50` 和 `n_window_infer=800` 计算 chunk，调用 `split`、
`pad_sequence`、布尔 mask 索引、Python `for` 循环和动态 `cu_seqlens`。随后每个
encoder layer 按实际长度构造 `[1,1,S,S]` block-diagonal mask。

这不是 RKNN 算子是否“列在支持表”就能解决的问题。即使导出成功，tracer 很容易把
该控制流固化为校准样本的长度/分块方式。建议把下列工作放到 CPU wrapper：

```text
PCM -> 128-mel -> 固定长度或长度桶的 chunk/pad -> encoder tensor + 静态 mask
```

NPU 图只接受静态 `[B, 128, T]`（或经过 CPU 前处理后的静态 tensor）和固定形状
mask。针对产品选择 2--4 个时长桶，而非一个“任意时长”图，例如短命令、10 s、30 s
和最大支持长度；超长音频由 CPU 切段，之后在 CPU 合并文本。

### B. 注意力与掩码

Audio encoder attention 使用 `MatMul -> scale -> Add(mask) -> Softmax(fp32) -> MatMul`。
这里有三项风险：

- 18 层都要生成/读取长序列注意力矩阵，NPU 内存峰值会随窗口长度平方增长；
- 源码使用 `torch.finfo(dtype).min` 作为 mask。TTS 已实测这个有限 FP32 极小值在
  RKNN FP16 路径可能下溢为 `-inf` 并污染 Softmax；ASR 导出也必须检查这一点；
- RKNN 支持表虽列有 `MatMul`、`Softmax`、`LayerNormalization` 和 `Gather`，但这不是
  “任意形状、任意布局均由 NPU 正确加速”的承诺。需要看 build 日志与板端 perf detail。

对 ASR 的实操修复与 TTS 一致：导出前把 purely-masking 的极小有限值替换为
FP16 安全的 `-10000.0`（前提是 ONNX Runtime 对原图/改图逐输出对齐），不要把
首次 NaN 误判为 INT8 量化问题。

### C. decoder KV cache，而非 Mamba state

ASR 的 decoder 在 `past_key_values` 非空后仅保留最后一个 token，说明它是标准
KV cache decode。没有 TTS 的 `conv_states` 或 `ssm_states`，状态面显著更小；但它仍
应避免在首版整图下沉：

- cache update/Concat/Scatter/Gather/Transpose 可能被 RKNN 重写为内部 layout-reorder；
- 每 token 把 cache 输入、输出和 logits 往返主存，容易吞掉 8 层、hidden 512 模型的
  计算收益；
- AR 输出的微小数值差异会改变 argmax / top-p 结果，随后递归分叉，不能只看首 token。

### D. 精度与量化

`model.safetensors` 是 BF16。首版应优先导出 FP16 静态子图来验证结构；再以真实音频
和真实 prefill/decode 状态建立 W8A8 或 W4A16 校准集。不要把任意 PyTorch 动态量化
ONNX 直接二次交给 RKNN：RKNN Toolkit2 2.3.2 的官方支持表明确把
`DynamicQuantizeLinear` 与 `MatMulInteger` 标为不支持。

对 ASR 而言，INT8 需分别验收 audio encoder 和 decoder；前者可用 W8A8 实测，后者
尤其需要保持 KV cache 的浮点边界。对于当前 RK3576，不能假设同一完整 W8A8 图可
同时保留 FP16 cache 接口。

## TTS 实测的可迁移结论

以下内容是本机真实 RK3576/TTS 实验的证据，不是泛化宣传结论。

| TTS 发现 | 对 ASR 是否适用 | 对 ASR 的行动 |
|---|---|---|
| Slow AR 的 Mamba conv/SSM state 接口被整图 W8A8 自动改成 INT8，非零状态数值失真 | **Mamba 部分不适用**；ASR 没有这些 state | 不以此作为 ASR 阻塞理由；但要单独确认 decoder KV cache 的 I/O dtype |
| 完整 Slow 图在 RK3576 实板的 cache reorder `rs-gather` 提交失败；W4A16 也未解决 | **高度相关**，因为 ASR decoder 也有 KV cache layout/reorder | 导出 decoder 时优先做 position 0/1/3/16 的实板测试；失败后把 KV update 留 CPU，不要只因可 build/load 就接入 |
| 完整 Slow FP16 连续递归 NPU 平均比 CPU 慢约 3.1 倍，且 teacher-forced 仍有位置相关误差 | **方法论直接适用**，绝对性能不可照搬 | ASR decoder 必须同时做 teacher-forced 和真递归；记录 token 一致性/文本 WER，而不仅是 cosine |
| 单层无状态 FFN FP16 数值可通过，但 50 次实板调用仍比 CPU 慢；T=64 也慢约 22% | **直接适用** | 不要把单层 Linear/FFN 做成每 token 独立 RKNN 调用；至少融合多层并计入 DMA 和 Runtime 调用 |
| TTS FP16 attention mask 的 `-3.4028235e38` 在 RKNN 路径产生 NaN；`-10000` 保持 ORT 语义且可避免该问题 | **直接适用** | ASR attention mask 导出前做这个有限值修复并回归比对 |
| build/load/init 成功不等于板端数值正确或性能收益 | **直接适用** | 验收门必须覆盖 ONNX、simulator、真实板、连续请求和端到端转写 |

最关键的区别是：**TTS Slow 的失败不能证明 ASR 一定失败，也绝不能证明“没有
Mamba，所以 ASR 整图 RKNN 一定能跑”。** ASR 消除了 SSM 状态复杂度，但保留了 attention
cache、动态预处理、长序列 mask 与自回归数值敏感性。

## 推荐部署设计

### 先验收的生产基线

```text
CPU: PCM/resample/mel/chunk 与动态长度处理
CPU (ONNX Runtime FP16/FP32): Audio encoder + bridge
CPU (ONNX Runtime FP16/FP32): Qwen2 decoder prefill/decode、KV cache、采样
CPU: detokenize/热词/标点与时间戳后处理
```

先在目标 RK3576 上测实时因子、峰值 RSS、长音频、中文/英文/混合语和热词。该基线是
后续 NPU 版本的唯一正确性基准。

### NPU 第一阶段：Audio encoder 静态 prefill

```text
CPU 固定分桶/补齐 + 生成静态 mask
-> NPU: 3 Conv + 18 audio-encoder 层 + audio MLP tower/projector（FP16）
-> CPU: 将 audio embedding 注入 decoder，并完成文本生成
```

理由：ASR 音频编码只运行一次，计算规模大，且没有跨 token 的 NPU state；它比 decoder
逐 token 调用更可能摊薄 NPU 调度与搬运开销。图可以按长度桶独立编译，避免动态控制流。

### NPU 第二阶段：仅在证据支持时迁移 decoder prefill

将给定 prompt/audio embedding 的 decoder prefill 静态化为若干长度桶；输出 hidden/
KV cache 到 CPU。不要在此阶段包含采样循环。若 prefill 只出现一次且 cache 输出很大，
该路径仍可能没有端到端收益，应以实测为准。

### 最后才评估 decode

若要尝试 NPU decode，图输入/输出应显式拆为当前 token、position 和每层 K/V cache；先
FP16、再重量化。每个长度桶至少测试：

```text
position 0（空 cache）
position 1（首个非零 cache）
position 3、16（累积状态）
20--100 token 真递归
```

只要发生 NPU 内部 Gather/layout 失败、token 分叉、端到端变慢或内存越限，即停止整图
decode 路线，采用“CPU decoder + NPU audio encoder”混合部署。

## 转换与验收清单

1. **锁定版本和目标。** 记录 checkpoint revision、Transformers/custom code、
   RKNN Toolkit2、Runtime、NPU driver/固件、`target_platform`。PC simulator 与板端
   结论分别记录。
2. **分图，而非直接 export 全模型。** 最少分为 `audio_encoder_bucket_T`、`audio_bridge`、
   `decoder_prefill_L`、`decoder_decode_L`；CPU wrapper 负责 chunk、mask、cache 和采样。
3. **先 FP16。** 固定真实 mel 和文本 prompt，逐输出比 ONNX Runtime：audio embedding、
   decoder logits、每层 KV delta。掩码有限值修复必须先证明 ORT 等价。
4. **只用真实校准样本。** 覆盖中文、英文、混合语、静音/噪声、短/长音频和不同 decoder
   position。随机正态样本不能代表 attention mask、cache 分布或 mel 动态范围。
5. **再量化并设置门槛。** 首个探索门：全部输出 finite、cosine >= 0.99、normalized L2
   <= 10%；生产门应更严格（建议关键 embedding/KV <= 1%），并要求贪心文本 token 一致。
   采样模式下用固定 seed、top-k/top-p 和最终 WER/CER 做回归。
6. **实板端到端验收。** 记录构建日志中的 CPU fallback、NPU perf detail、首调用与稳态
   延迟、CPU->NPU->CPU 搬运、RSS/CMA/NPU 内存和 RTF。仅报告全链路速度，不能用单
   MatMul 或 simulator 延迟替代。

## 当前判定

| 项目 | 判定 |
|---|---|
| ASR 是否有 Mamba/SSM | 否，已由 checkpoint config 与 custom code 交叉确认 |
| 是否受 TTS Mamba state 直接阻塞 | 否 |
| 是否仍有 cache/Gather/递归风险 | 是，decoder KV cache 路径需要单独实测 |
| 是否建议完整动态 ASR 一图 RKNN | 不建议 |
| 是否值得优先尝试 NPU | 是，优先固定长度的 audio encoder prefill FP16 |
| 是否可据现有证据宣称 ASR RKNN 可部署 | 不可以；尚未对 ASR 图运行转换或 RK3576 实板验证 |

## 本地证据索引

- [TTS RK3576 总体部署调研](C:/Users/Administrator/Documents/ChatGPT/TTS/Audio8_TTS/docs/rk3576-rknn-deployment-research.zh-CN.md)
- [完整 Slow 连续递归探针](C:/Users/Administrator/Documents/ChatGPT/TTS/Audio8_TTS/rk3576/slow_continuous_probe_report.zh-CN.md)
- [无状态 FFN 子图实测](C:/Users/Administrator/Documents/ChatGPT/TTS/Audio8_TTS/rk3576/slow_stateless_subgraph_findings.zh-CN.md)
- [完整 Slow W4A16 实板失败记录](C:/Users/Administrator/Documents/ChatGPT/TTS/Audio8_TTS/rk3576/slow_w4a16_validation.md)
- [RKNN Toolkit2 2.3.2 ONNX 算子表](C:/Users/Administrator/Documents/ChatGPT/TTS/rknn-toolkit2-upstream/doc/RKNNToolKit2_OP_Support-2.3.2.md)
