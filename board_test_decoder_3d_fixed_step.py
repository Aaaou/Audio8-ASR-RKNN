"""One full Qwen2 decode step: NPU blocks/head, CPU-only KV append and argmax."""
import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
def rss():
 for x in Path('/proc/self/status').read_text().splitlines():
  if x.startswith('VmRSS:'):return int(x.split()[1])/1024
def rel(x,y):return float(np.linalg.norm(x.astype(np.float64)-y.astype(np.float64))/np.linalg.norm(y.astype(np.float64)))
def stat(a):
 a=np.array(a)*1e3;return {'mean':float(a.mean()),'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95))}
a=argparse.ArgumentParser();a.add_argument('--dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);z=a.parse_args();d=z.dir
before=rss();kv=[];blocks=[];heads=[]
try:
 for i in range(8):
  k,b=RKNNLite(verbose=False),RKNNLite(verbose=False);assert k.load_rknn(str(d/f'layer{i}/kv_fp16.rknn'))==0 and b.load_rknn(str(d/f'layer{i}/block_fp16.rknn'))==0;assert k.init_runtime()==0 and b.init_runtime()==0;kv.append(k);blocks.append(b)
 for i in range(8):
  h=RKNNLite(verbose=False);assert h.load_rknn(str(d/f'head_shards/shard{i:02d}/head_fp16.rknn'))==0 and h.init_runtime()==0;heads.append(h)
 init=rss();x=np.load(d/'layer0/hidden.npy').astype('float32');co=np.load(d/'layer0/cos.npy').astype('float32');si=np.load(d/'layer0/sin.npy').astype('float32');layer_rows=[];kt=[];bt=[];ct=[]
 for i in range(8):
  ld=d/f'layer{i}';fk=np.load(ld/'full_k.npy').astype('float32');fv=np.load(ld/'full_v.npy').astype('float32');t=time.perf_counter();dk,dv=kv[i].inference(inputs=[x,co,si]);kt.append(time.perf_counter()-t);t=time.perf_counter();fk[:,110:111,:]=dk;fv[:,110:111,:]=dv;ct.append(time.perf_counter()-t);t=time.perf_counter();x=blocks[i].inference(inputs=[x,fk,fv,co,si,np.load(ld/'mask.npy').astype('float32')])[0];bt.append(time.perf_counter()-t);layer_rows.append({'layer':i,'key_l2':rel(dk,np.load(ld/'key_delta_reference.npy')),'value_l2':rel(dv,np.load(ld/'value_delta_reference.npy')),'hidden_l2':rel(x,np.load(ld/'hidden_reference.npy'))})
 ht=[];pieces=[]
 for h in heads:
  t=time.perf_counter();pieces.append(h.inference(inputs=[x])[0]);ht.append(time.perf_counter()-t)
 logits=np.concatenate(pieces,axis=-1);ref=np.load(d/'head/logits_reference.npy');out={'accuracy':{'per_layer':layer_rows,'logits_l2':rel(logits,ref),'npu_argmax':int(logits.argmax(-1)),'reference_argmax':int(ref.argmax(-1))},'latency_ms':{'kv_npu_8_layers':stat(kt),'cpu_cache_write_8_layers':stat(ct),'block_npu_8_layers':stat(bt),'head_npu_8_shards':stat(ht),'total_npu_compute':float((sum(kt)+sum(bt)+sum(ht))*1e3),'total_cpu_kv_write':float(sum(ct)*1e3)},'memory':{'cache_bytes_fp32_kv':int(sum(np.load(d/f'layer{i}/full_k.npy').nbytes+np.load(d/f'layer{i}/full_v.npy').nbytes for i in range(8))),'rss_before_mib':before,'rss_after_init_mib':init,'rss_after_mib':rss()}}
 z.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
finally:
 for r in kv+blocks+heads:r.release()
