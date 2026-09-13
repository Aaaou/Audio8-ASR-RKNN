"""Strict multi-token CPU-cache/NPU-Qwen2 schedule acceptance on RK3576."""
import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
def rss():
 for x in Path('/proc/self/status').read_text().splitlines():
  if x.startswith('VmRSS:'):return int(x.split()[1])/1024
def l2(x,y):return float(np.linalg.norm(x.astype(np.float64)-y.astype(np.float64))/np.linalg.norm(y.astype(np.float64)))
def s(v):
 v=np.array(v)*1000;return {'mean':float(v.mean()),'p50':float(np.percentile(v,50)),'p95':float(np.percentile(v,95))}
a=argparse.ArgumentParser();a.add_argument('--dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);z=a.parse_args();d=z.dir;meta=json.loads((d/'schedule_refs/manifest.json').read_text());initial=meta['initial_length'];before=rss();kvs=[];blocks=[];heads=[]
try:
 for i in range(8):
  k,b=RKNNLite(verbose=False),RKNNLite(verbose=False);assert k.load_rknn(str(d/f'layer{i}/kv_fp16.rknn'))==0 and b.load_rknn(str(d/f'layer{i}/block_fp16.rknn'))==0;assert k.init_runtime()==0 and b.init_runtime()==0;kvs.append(k);blocks.append(b)
 for i in range(8):
  h=RKNNLite(verbose=False);assert h.load_rknn(str(d/f'head_shards/shard{i:02d}/head_fp16.rknn'))==0 and h.init_runtime()==0;heads.append(h)
 after_init=rss();cache_k=[];cache_v=[]
 for i in range(8):
  k=np.load(d/f'layer{i}/full_k.npy').astype('float32');v=np.load(d/f'layer{i}/full_v.npy').astype('float32');k[:,initial:,:]=0;v[:,initial:,:]=0;cache_k.append(k);cache_v.append(v)
 rows=[];allkv=[];allblock=[];allhead=[];allwrite=[]
 # The next input is selected from the actual NPU argmax below.  In a product
 # build this is a CPU embedding-table lookup; these six saved rows are the
 # minimal table needed for this strict, deterministic trajectory.
 emb={}; ropes=[]
 for step,tok in enumerate(meta['input_tokens']):
  ref=d/f'schedule_refs/step{step}';emb[tok]=np.load(ref/'hidden.npy').astype('float32');ropes.append((np.load(ref/'cos.npy').astype('float32'),np.load(ref/'sin.npy').astype('float32')))
 x=emb[meta['input_tokens'][0]]
 for step,expected in enumerate(meta['output_tokens']):
  ref=d/f'schedule_refs/step{step}';co,si=ropes[step];pos=initial+step;mask=np.full((8,1,128),-10000.,dtype='float32');mask[:,:,:pos+1]=0;layers=[];input_l2=l2(x,np.load(ref/'hidden.npy'))
  for i in range(8):
   t=time.perf_counter();dk,dv=kvs[i].inference(inputs=[x,co,si]);allkv.append(time.perf_counter()-t);t=time.perf_counter();cache_k[i][:,pos:pos+1,:]=dk;cache_v[i][:,pos:pos+1,:]=dv;allwrite.append(time.perf_counter()-t);t=time.perf_counter();x=blocks[i].inference(inputs=[x,cache_k[i],cache_v[i],co,si,mask])[0];allblock.append(time.perf_counter()-t);layers.append({'layer':i,'key_l2':l2(dk,np.load(ref/f'layer{i}_key.npy')),'value_l2':l2(dv,np.load(ref/f'layer{i}_value.npy')),'hidden_l2':l2(x,np.load(ref/f'layer{i}_hidden.npy'))})
  pieces=[]
  for h in heads:t=time.perf_counter();pieces.append(h.inference(inputs=[x])[0]);allhead.append(time.perf_counter()-t)
  got=int(np.concatenate(pieces,axis=-1).argmax(-1).reshape(-1)[0]);rows.append({'step':step,'input_token':meta['input_tokens'][step],'input_embedding_l2':input_l2,'expected_token':expected,'npu_token':got,'token_match':got==expected,'layers':layers})
  if step+1 < len(meta['output_tokens']):
   if got not in emb: raise RuntimeError(f'NPU emitted {got}, not present in deterministic embedding lookup table')
   x=emb[got]
 out={'contract':{'CPU':'cache buffer allocation, in-place K/V write, token argmax','NPU':'8 K/V projections + 8 decoder blocks + 8 LM-head shards per token','cache_layout':'8 layers x K/V [8,128,64] FP32'},'accuracy':{'steps':rows,'all_token_match':all(x['token_match'] for x in rows),'max_key_l2':max(q['key_l2'] for x in rows for q in x['layers']),'max_value_l2':max(q['value_l2'] for x in rows for q in x['layers']),'max_hidden_l2':max(q['hidden_l2'] for x in rows for q in x['layers'])},'latency_ms':{'kv_npu_call':s(allkv),'block_npu_call':s(allblock),'head_shard_npu_call':s(allhead),'cpu_cache_write':s(allwrite),'per_token_npu_mean':float((sum(allkv)+sum(allblock)+sum(allhead))/len(rows)*1000),'per_token_cpu_cache_write_mean':float(sum(allwrite)/len(rows)*1000)},'memory':{'cpu_kv_buffers_bytes':int(sum(x.nbytes+y.nbytes for x,y in zip(cache_k,cache_v))),'rss_before_init_mib':before,'rss_after_init_mib':after_init,'rss_after_runs_mib':rss()}}
 z.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
finally:
 for r in kvs+blocks+heads:r.release()
