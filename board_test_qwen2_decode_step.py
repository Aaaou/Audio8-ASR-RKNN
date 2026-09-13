"""Profile and validate fixed-S KV-cache decode step on RK3576."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
def rss_mib():
 for line in Path("/proc/self/status").read_text().splitlines():
  if line.startswith("VmRSS:"): return int(line.split()[1])/1024
 return None
def l2(a,b): return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/np.linalg.norm(b.astype(np.float64)))
p=argparse.ArgumentParser();p.add_argument("--dir",type=Path,required=True);p.add_argument("--runs",type=int,default=20);p.add_argument("--output",type=Path,required=True);p.add_argument("--input-dtype",choices=("float32","float16"),default="float32");a=p.parse_args();m=json.loads((a.dir/"manifest.json").read_text())
input_dtype=np.float16 if a.input_dtype=="float16" else np.float32
ins=[np.load(a.dir/(x+".npy")).astype(input_dtype) for x in m["inputs"]];refs=[np.load(a.dir/(x+"_reference.npy")).astype(np.float32) for x in m["outputs"]]
r=RKNNLite(verbose=False)
try:
 print("rknn_api", [x for x in dir(r) if "input" in x.lower() or "query" in x.lower()])
 before=rss_mib();assert r.load_rknn(str(a.dir/"qwen2_decode_step_s110_delta_fp16.rknn"))==0 and r.init_runtime()==0; after_init=rss_mib();r.inference(inputs=ins)
 latency=[];concat=[];actual=None
 for _ in range(a.runs):
  t=time.perf_counter();actual=[np.asarray(x) for x in r.inference(inputs=ins)];latency.append(time.perf_counter()-t)
  t=time.perf_counter();updated=[np.concatenate((old,new),axis=-2) for old,new in zip(ins[1:],actual[1:])];concat.append(time.perf_counter()-t)
 payload={"cache_length":m["cache_length"],"runs":a.runs,"rss_mib":{"before_rknn":before,"after_rknn_init":after_init,"after_runs":rss_mib()},"io_bytes":{"token_embedding":int(ins[0].nbytes),"past_cache_input":int(sum(x.nbytes for x in ins[1:])),"logits_output":int(actual[0].nbytes),"kv_delta_output":int(sum(x.nbytes for x in actual[1:])),"cpu_updated_cache":int(sum(x.nbytes for x in updated))},"dtypes":{"inputs":sorted(set(str(x.dtype) for x in ins)),"outputs":sorted(set(str(x.dtype) for x in actual))},"latency_ms":{"rknn_end_to_end_mean":float(np.mean(latency)*1e3),"rknn_end_to_end_p50":float(np.percentile(latency,50)*1e3),"rknn_end_to_end_p95":float(np.percentile(latency,95)*1e3),"cpu_cache_concat_mean":float(np.mean(concat)*1e3),"cpu_cache_concat_p95":float(np.percentile(concat,95)*1e3)},"logits":{"normalized_l2":l2(actual[0],refs[0]),"reference_argmax":int(refs[0].argmax()),"rknn_argmax":int(actual[0].argmax()),"argmax_match":bool(actual[0].argmax()==refs[0].argmax())},"kv_delta":{"max_normalized_l2":max(l2(x,y) for x,y in zip(actual[1:],refs[1:])),"mean_normalized_l2":float(np.mean([l2(x,y) for x,y in zip(actual[1:],refs[1:])]))}}
 a.output.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
finally:r.release()
