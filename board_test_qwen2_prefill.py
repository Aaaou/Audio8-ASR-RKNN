"""Board-side fixed Qwen2 prefill logits test (no KV cache)."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite

parser = argparse.ArgumentParser()
parser.add_argument("--rknn", type=Path, required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
x, ref = np.load(args.input).astype(np.float32), np.load(args.reference).astype(np.float32)
rknn = RKNNLite(verbose=False)
try:
    if rknn.load_rknn(str(args.rknn)) != 0 or rknn.init_runtime() != 0: raise RuntimeError("RKNN init failed")
    rknn.inference(inputs=[x])
    times=[]
    for _ in range(args.runs):
        start=time.perf_counter(); actual=np.asarray(rknn.inference(inputs=[x])[0], dtype=np.float32); times.append(time.perf_counter()-start)
    delta=actual.astype(np.float64)-ref.astype(np.float64)
    payload={"input_shape":list(x.shape),"output_shape":list(actual.shape),"runs":args.runs,
      "normalized_l2":float(np.linalg.norm(delta)/np.linalg.norm(ref.astype(np.float64))), "max_abs":float(np.abs(delta).max()),
      "cosine":float(np.sum(actual.astype(np.float64)*ref.astype(np.float64))/(np.linalg.norm(actual.astype(np.float64))*np.linalg.norm(ref.astype(np.float64)))),
      "reference_argmax":int(ref.argmax()),"rknn_argmax":int(actual.argmax()),
      "latency_ms":{"mean":float(np.mean(times)*1e3),"p50":float(np.percentile(times,50)*1e3),"min":float(np.min(times)*1e3),"max":float(np.max(times)*1e3)}}
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))
finally: rknn.release()
