"""Full RKNN ASR with CPU-owned KV cache that grows across 128/256/512 buckets."""
import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
def load(p):
 r=RKNNLite(verbose=False);assert r.load_rknn(str(p))==0 and r.init_runtime()==0;return r
def rss():
 for x in Path('/proc/self/status').read_text().splitlines():
  if x.startswith('VmRSS:'):return int(x.split()[1])/1024
def st(x):
 a=np.asarray(x)*1000;return {'mean':float(a.mean()),'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95))}
a=argparse.ArgumentParser();a.add_argument('--input-dir',type=Path,required=True);a.add_argument('--prefill-rknn',type=Path,required=True);a.add_argument('--asset-root',type=Path,help='release root; uses legacy board paths when omitted');a.add_argument('--adapter-rknn',type=Path);a.add_argument('--model',type=Path,required=True);a.add_argument('--max-new',type=int,default=400);a.add_argument('--ignore-eos',action='store_true',help='scheduler stress mode: continue greedy decoding after EOS');a.add_argument('--output',type=Path,required=True);z=a.parse_args();inp=z.input_dir;cfg=json.loads((z.model/'config.json').read_text());eos={cfg['eos_token_id']} if isinstance(cfg['eos_token_id'],int) else set(cfg['eos_token_id']);allr=[];before=rss();total=time.perf_counter()
root=z.asset_root
def asset(release,legacy): return root/release if root else Path(legacy)
adapter=z.adapter_rknn or asset('adapter/audio_adapter_h104_t100_fp16.rknn','/root/audio8-asr/audio_adapter_h104_t100/audio_adapter_h104_t100_fp16.rknn')
try:
 enc=load(asset('encoder/audio_encoder_f800_fp16.rknn','/root/audio8-asr/encoder_f800_v2/audio_encoder_f800_fp16.rknn'));adp=load(adapter);pre=load(z.prefill_rknn);allr=[enc,adp,pre]
 mel=np.load(inp/'input_features.npy').astype('float32');ids=np.load(inp/'input_ids.npy').reshape(-1);audio_pos=np.load(inp/'audio_positions.npy').reshape(-1);table=np.load(asset('token_embeddings_fp32.npy','/root/audio8-asr/decoder_3d_fixed_s256/token_embeddings_fp32.npy'),mmap_mode='r');t=time.perf_counter();e=enc.inference(inputs=[mel])[0];enc_s=time.perf_counter()-t;t=time.perf_counter();audio=adp.inference(inputs=[e])[0];adp_s=time.perf_counter()-t
 if len(audio) != len(audio_pos): raise RuntimeError(f'adapter tokens {len(audio)} != audio placeholders {len(audio_pos)}')
 embed=np.asarray(table[ids],np.float32)[None,:,:];embed[0,audio_pos,:]=audio;t=time.perf_counter();po=pre.inference(inputs=[embed]);pre_s=time.perf_counter()-t
 for r in allr:r.release()
 allr=[];kvs=[load(asset(f'decoder/kv/layer{i}/kv_fp16.rknn',f'/root/audio8-asr/decoder_3d_fixed_s128/layer{i}/kv_fp16.rknn')) for i in range(8)];heads=[load(asset(f'decoder/head_shards/shard{i:02d}/head_fp16.rknn',f'/root/audio8-asr/decoder_3d_fixed_s128/head_shards/shard{i:02d}/head_fp16.rknn')) for i in range(8)];allr=kvs+heads;blocks={};capacity=128
 def get_blocks(c):
  if c not in blocks:
   blocks[c]=[load(asset(f'decoder/block_s{c}/layer{i}/block_fp16.rknn',f'/root/audio8-asr/decoder_3d_fixed_s{c}/layer{i}/block_fp16.rknn')) for i in range(8)];allr.extend(blocks[c])
  return blocks[c]
 initial=po[1].shape[-2];ck=[];cv=[]
 for i in range(8):
  k=np.zeros((8,capacity,64),np.float32);v=np.zeros_like(k);k[:,:initial]=po[1+2*i].reshape(8,initial,64);v[:,:initial]=po[2+2*i].reshape(8,initial,64);ck.append(k);cv.append(v)
 inv=np.load(inp/'rotary_inv_freq.npy').astype('float32');token=int(np.asarray(po[0]).argmax());tokens=[token];switches=[];kv_t=[];block_t=[];head_t=[];write_t=[]
 for step in range(z.max_new):
  if step and token in eos and not z.ignore_eos:break
  pos=initial+step
  if pos>=capacity:
   next_c=256 if capacity==128 else 512 if capacity==256 else None
   if next_c is None:raise RuntimeError(f'no bucket above {capacity}')
   t=time.perf_counter();newk=[np.pad(x,((0,0),(0,next_c-capacity),(0,0))) for x in ck];newv=[np.pad(x,((0,0),(0,next_c-capacity),(0,0))) for x in cv];mig=time.perf_counter()-t;switches.append({'at_position':pos,'from':capacity,'to':next_c,'cpu_cache_copy_ms':mig*1000});ck,cv,capacity=newk,newv,next_c
  bl=get_blocks(capacity);x=np.asarray(table[token:token+1],np.float32).reshape(1,1,512);f=np.concatenate((inv*pos,inv*pos))[None,None,:];co=np.cos(f).astype('float32');si=np.sin(f).astype('float32');mask=np.full((8,1,capacity),-10000,np.float32);mask[:,:,:pos+1]=0
  for i in range(8):
   t=time.perf_counter();dk,dv=kvs[i].inference(inputs=[x,co,si]);kv_t.append(time.perf_counter()-t);t=time.perf_counter();ck[i][:,pos:pos+1]=dk;cv[i][:,pos:pos+1]=dv;write_t.append(time.perf_counter()-t);t=time.perf_counter();x=bl[i].inference(inputs=[x,ck[i],cv[i],co,si,mask])[0];block_t.append(time.perf_counter()-t)
  pieces=[]
  for h in heads:t=time.perf_counter();pieces.append(h.inference(inputs=[x])[0]);head_t.append(time.perf_counter()-t)
  token=int(np.concatenate(pieces,-1).argmax());tokens.append(token)
 out={'complete_to_eos':tokens[-1] in eos,'scheduler_stress_ignore_eos':z.ignore_eos,'eos_ids':list(eos),'tokens':tokens,'generated_tokens_excluding_prefill_token':len(tokens)-1,'bucket_switches':switches,'final_bucket':capacity,'latency_ms':{'encoder':enc_s*1000,'adapter':adp_s*1000,'prefill':pre_s*1000,'kv_npu':st(kv_t),'block_npu':st(block_t),'head_npu':st(head_t),'cpu_kv_write':st(write_t),'decoder_npu_total':float((sum(kv_t)+sum(block_t)+sum(head_t))*1000),'decoder_cpu_kv_total':float(sum(write_t)*1000),'steady_end_to_end_total':float((enc_s+adp_s+pre_s+sum(kv_t)+sum(block_t)+sum(head_t)+sum(write_t)+sum(x['cpu_cache_copy_ms']/1000 for x in switches))*1000),'process_wall_total':float((time.perf_counter()-total)*1000)},'rss_mib':{'before':before,'after':rss()}};z.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
finally:
 for r in allr:r.release()
