# CPU-owned KV cache strategy

The first DynamicCache export was intentionally stopped after layer-0 debug
showed the first failure at NPU cache Concat. It is not the CPU-owned strategy.

The implementation to use next is an explicit read-only-cache graph:

```text
inputs: hidden [1,1,512], full_cache_k [1,8,S+1,64], full_cache_v [1,8,S+1,64]
NPU: q projection, rotary, attention against the already CPU-concatenated cache,
     output projection, MLP, logits, and current K/V delta
CPU: compute/obtain delta, concat old cache + delta, pass full cache next step
```

There must be no `DynamicCache.update`, `ScatterND`, or cache `Concat` in the
NPU graph. The first production candidate will use a two-graph step if needed:
one projection graph for current K/V and one read-only attention/block graph.
This adds a measurable synchronization boundary but makes ownership explicit.

Current evidence requiring this rewrite:

```text
layer-0 new K/V projection: L2 0.0645% / 0.0917%
layer-0 NPU Concat:          L2 141.31% / 140.04%
```

Until the explicit read-only graph passes layer-0 logits and cache comparisons,
no S=111 multi-step graph should be built.
