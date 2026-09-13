# Silero VAD 接入、BM 与长音频体感评估

## 结论

Silero VAD 已接入长音频调度器：`Silero VAD → 有声段 → 8/16/30 秒静态
Audio8 profile → 每段独立 RKNN ASR → 按时间轴显示`。它只决定**何时提交一段
音频给 ASR**，不改 Audio8 的 encoder、prefill、decoder 或 KV cache 语义。

本次 30.000 秒输入在 RK3576 上由 Silero 切成 5 段（4×s8、1×s16）。所有段均
走到 EOS；在完全相同的物理补零、`audio_padding=max_length` 和 profile 契约下，
与原始 PyTorch CPU FP32 逐 token 及 EOS 完全相同，故相对验收 **CER=0%、WER=0%**。
这不是带人工标注语料的绝对识别率测试。

Silero 本身运行在 **CPU ONNX Runtime**，不使用 RKNN 或 NPU；Audio8 神经网络
仍由 RKNN 图在 RKNPU 上执行。

## 接入与可复现方式

实现位于 `long_audio_segmenter.py`，新增 `--vad-backend silero` 和
`--silero-threshold`。已测版本及参数：

| 项目 | 值 |
|---|---|
| VAD | Silero VAD 6.2.1，官方 Python 包 |
| 执行后端 | ONNX Runtime CPU |
| 采样率 | 16 kHz、单声道 |
| threshold | 0.50 |
| 最短有声段 | 250 ms |
| 最短静音（分段合并） | 300 ms |
| RK3576 环境 | `torch 2.2.0`、`torchaudio 2.2.0`、`onnxruntime`、`silero-vad 6.2.1` |

```bash
python long_audio_segmenter.py --audio INPUT_16K_MONO.wav --output-dir plan \
  --vad-backend silero --silero-threshold 0.5 \
  --preferred-seconds 16 --maximum-seconds 24 --overlap-seconds 0.6
python prepare_deployment_bucket_inputs.py --model MODEL_DIR --plan-dir plan --output-dir inputs
python3 board_run_long_audio.py --asset-root RELEASE_ROOT \
  --inputs-manifest inputs/inputs-manifest.json \
  --token-embeddings RELEASE_ROOT/token_embeddings_fp32.npy --output result.json
```

当前实现的 VAD 是离线文件规划：它先扫完整文件、再顺序提交段。若应用需要麦克风
级低延迟，应把同一个 Silero 状态机按音频块持续喂入，段结束即提交；这属于应用层
流式调度，不代表 Audio8 checkpoint 获得跨段原生 KV streaming。

## VAD 微基准

下表为 x86 端 ONNX Runtime CPU 的 10 次热态重复（不含读取 WAV）；Silero 的一次
冷加载为 127.96 ms。energy 是此前的轻量能量门限基线（-60 dB、30 ms frame），用于
比较开销和切分，不代表同等检测能力。

| 原始长度 | energy mean | Silero mean | Silero P95 | Silero 段数 | Silero 有声时长 |
|---:|---:|---:|---:|---:|---:|
| 2 s | 0.215 ms | 13.590 ms | 16.492 ms | 1 | 1.486 s |
| 4 s | 0.250 ms | 21.074 ms | 23.520 ms | 1 | 3.486 s |
| 6 s | 0.228 ms | 30.400 ms | 35.206 ms | 1 | 5.020 s |
| 30 s | 0.876 ms | 149.815 ms | 158.937 ms | 5 | 25.484 s |

RK3576 上也实际运行了同一官方 ONNX 路径（10 次热态、30 秒）：Silero mean
**1463.668 ms**、P50 **1444.799 ms**、P95 **1554.908 ms**、冷加载 **347.111 ms**，
即 VAD 稳态 RTF = `1.463668 / 30` = **0.0488**。这比 x86 慢，但在完整 ASR
链路中仍远小于每段 RKNN 图映射/释放时间。

在该测试文件上，Silero 的边界为 0.546–2.686、2.978–5.054、5.378–10.046、
11.490–18.718、19.746–29.118 秒，共保留 25.484 秒有声内容；没有触发强制切段
或 overlap。VAD 不同会改变送入 ASR 的音频内容，因此不能把两个 VAD 方案的 ASR
耗时差异归因于 VAD 算法运行速度。

## 30 秒完整 RK3576 应用链路

`neural` 是 encoder、adapter、prefill、所有 decoder NPU 调用及 CPU KV 写入；
`task wall` 还包括为适配 4 GiB 板端内存而进行的 RKNN frontend/decoder 分阶段
map/init/release。两者均不含 VAD、WAV 解码、重采样及 host processor 准备。

| 段 | 时间范围 | profile | token（含 EOS） | neural | task wall | 关键 KV / RSS |
|---:|---|---|---:|---:|---:|---|
| 0 | 0.546–2.686 s | s8 | 10 | 3705.40 ms | 13264.78 ms | S128；1407.73 / 1015.82 MiB |
| 1 | 2.978–5.054 s | s8 | 7 | 3488.89 ms | 10301.63 ms | S128；1430.48 / 1019.81 MiB |
| 2 | 5.378–10.046 s | s8 | 16 | 3927.73 ms | 11022.62 ms | S128；1560.98 / 1037.11 MiB |
| 3 | 11.490–18.718 s | s8 | 27 | 9058.73 ms | 15996.02 ms | S128→S256，copy 31.07 ms；1628.47 / 1067.15 MiB |
| 4 | 19.746–29.118 s | s16 | 33 | 7031.65 ms | 16212.52 ms | S256；1781.46 / 1070.79 MiB |
| **合计** | **原始 30.000 s** | **4×s8 + 1×s16** | **93** | **27212.41 ms** | **67381.31 ms worker wall** | **全部 EOS；token/EOS/CER/WER 全通过** |

段表中的前后两个 RSS 数字分别为 frontend、decoder stage 的峰值；全轮峰值为
**1781.46 MiB frontend**、**1070.79 MiB decoder**，不是二者同时常驻。所有已加载的
Audio8 图仍报告 `RKNPU f2`，CPU 不重算 decoder；CPU KV 只维护外置缓存和 bucket
扩容拷贝。

`neural` RTF = `27.212 / 30` = **0.907**；当前完整 worker RTF =
`67.381 / 30` = **2.246**。差距主要是为规避 4 GiB OOM 而反复映射/释放 RKNN 图，
并非 Silero VAD。若仅看离线文件模式的“首段文字出现”，本样本为完整 VAD 扫描
1.464 秒后加第 0 段 13.265 秒，约 **14.73 秒**（模型/VAD 已热态，不含文件读取）。

## 原生 Audio8 各长度已测数据

下表是此前的原生单段 FP16 RKNN 端到端 `neural` 实测，且全部与同一输入契约下的
FP32 token/EOS 一致。2/4/6/8 秒使用相应 T25/T50/T75/T100 静态输入；30 秒使用官方
最长输入。它们不能与上表按 padding 的 s8/s16 段简单拼成线性曲线，因为 decoder
耗时还受输出 token 数影响。

| 音频长度 | RKNN neural E2E | neural RTF |
|---:|---:|---:|
| 2 s | 1231.49 ms | 0.616 |
| 4 s | 3371.87 ms | 0.843 |
| 6 s | 4116.86 ms | 0.686 |
| 8 s | 4344.29 ms | 0.543 |
| 30 s（官方最长输入） | 22121.31 ms | 0.737 |

## 15 分钟预估与体感边界

这是以本页**真实 30 秒 Silero 分段样本**做的 30 倍线性外推，而非尚未执行的 15 分钟
实跑。假设语音/静音比例、5 段/30 秒的 profile 分布、token 密度、NPU 频率、温度和
内存状态都相近；不含音频解码、重采样、processor、磁盘/网络 I/O。超长连续语音若
触发 maximum 24 秒的强制切段和 0.6 秒 overlap，会增加计算，故实际应视为区间而非
保证值。

| 15 分钟（900 s）口径 | 推导 | 预估 |
|---|---:|---:|
| ASR neural | 27.212 s × 30 | **816.37 s（13 分 36.37 秒）** |
| 当前 memory-safe worker wall | 67.381 s × 30 | **2021.44 s（33 分 41.44 秒）** |
| 板端 Silero VAD 热态 | 1.463668 s × 30 | **43.91 s** |
| 板端 VAD + ASR neural | 上两项相加 | **860.28 s（14 分 20.28 秒）** |
| 板端 VAD + 当前 worker wall | 上两项相加 | **2065.35 s（34 分 25.35 秒）** |
| x86 Silero VAD 热态（若前处理不在板端） | 0.149815 s × 30 | **4.49 s** |

板端 VAD 的一次冷加载再增加约 0.347 秒；x86 冷加载增加约 0.128 秒。当前离线实施中，
15 分钟文件必须先完成约 44 秒的板端 VAD 扫描才开始第一段 ASR。改成流式 VAD 后，
用户可在第一段结束后立即看到该段文字：就本样本的第一个有声段而言，是收齐约 2.686
秒音频后再加约 13.265 秒 ASR，而非等待整段 15 分钟录音结束。

因此，当前应用可**逐段显示**、精度已验收，但在现有 4 GiB 的分阶段图生命周期下，
完整离线任务 wall time 尚不满足实时。下一轮真正改善体感的方向是降低图 map/release
开销或在可承受内存内保持必要图常驻；不是用 VAD 替代该项优化。
