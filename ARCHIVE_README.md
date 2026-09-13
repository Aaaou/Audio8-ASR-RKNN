# Audio8-ASR RK3576 experiment archive

This directory is an experiment archive for `Edge0/Audio8-ASR-0.1B` on
Firefly ROC-RK3576. It contains the research notes, RKNN conversion/test
scripts, and small JSON benchmark outputs produced on 2026-09-13.

Excluded deliberately: model weights, WAV files, `.rknn` binaries, and RKNN
Toolkit temporary ONNX outputs. They are either redistributable model assets,
large reproducible build products, or locally supplied test material.

The verified hybrid execution boundary is:

```text
CPU audio decode/resample/mel -> RK3576 NPU FP16 RKNN audio encoder
-> CPU audio tower/projector/Qwen2 autoregressive decoder -> text
```

The FP16 encoder's RKNN runtime target is `RKNPU f2` / `rk3576`. W8A8 and
W4A16 full-encoder candidates are retained as benchmark records only and are
not deployment candidates due to high encoder-output error.
