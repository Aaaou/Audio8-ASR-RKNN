# Next repair: external cache as 3D head-batched tensors

The four-dimensional cache boundary `[B,H,S,D]` is corrupted by the RKNN Lite
layout path when used by the attention block. The next decoder graph contract
must use contiguous rank-3 tensors:

```text
K/V input: [H,S,D] = [8,111,64]
Q:           [H,1,D] = [8,1,64]
score:       [H,1,S]
V output:    [H,1,D]
```

CPU still owns cache and appends along axis 1. NPU receives only rank-3 cache
inputs, which avoids the problematic RKNN 4D NHWC/NCHW boundary conversion.
The graph will restore `[1,1,512]` only after head attention is complete.
