"""Real Audio8 RKNN pipeline: frontend -> prefill KV -> autonomous decode to EOS."""
import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
def load(path):
 r=RKNNLite(verbose=False);assert r.load_rknn(str(path))==0 and r.init_runtime()==0;return r
def rss():
 for x in Path('/proc/self/status').read_text().splitlines():
  if x.startswith('VmRSS:'):return int(x.split()[1])/1024
def stats(x):
 x=np.asarray(x)*1000;return {'mean':float(x.mean()),'p50':float(np.percentile(x,50)),'p95':float(np.percentile(x,95))}
a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--model',type=Path,required=True);a.add_argument('--max-new',type=int,default=128);a.add_argument('--output',type=Path,required=True);z=a.parse_args();d=z.root;cfg=json.loads((z.model/'config.json').read_text());eos=cfg.get('eos_token_id');eos=set(eos if isinstance(eos,list) else [eos]);start=time.perf_counter();before=rss();allr=[]
try:
 enc=load('/root/audio8-asr/encoder_f800_v2/audio_encoder_f800_fp16.rknn');adp=load('/root/audio8-asr/audio_adapter_h104_t100/audio_adapter_h104_t100_fp16.rknn');pre=load('/root/audio8-asr/prefill_kv_s110/prefill_kv_s110_fp16.rknn');allr += [enc,adp,pre]
 mel=np.load(d/'input_features.npy').astype('float32');ids=np.load(d/'input_ids.npy').reshape(-1);pos=np.load(d/'audio_positions.npy').reshape(-1)
 t=time.perf_counter();raw=enc.inference(inputs=[mel])[0];encoder_s=time.perf_counter()-t;t=time.perf_counter();audio=adp.inference(inputs=[raw])[0];adapter_s=time.perf_counter()-t
 table=np.load(d/'token_embeddings_fp32.npy',mmap_mode='r');template=np.asarray(table[ids],dtype=np.float32)[None,:,:]
 template[0,pos,:]=audio;t=time.perf_counter();preout=pre.inference(inputs=[template]);prefill_s=time.perf_counter()-t
 for r in [enc,adp,pre]:r.release()
 allr=[];kvs=[];blocks=[];heads=[]
 for i in range(8):kvs.append(load('/root/audio8-asr/decoder_3d_fixed_s128/layer%d/kv_fp16.rknn'%i));blocks.append(load(d/f'layer{i}/block_fp16.rknn'))
 for i in range(8):heads.append(load('/root/audio8-asr/decoder_3d_fixed_s128/head_shards/shard%02d/head_fp16.rknn'%i))
 allr=kvs+blocks+heads;smax=256;initial=preout[1].shape[-2];cache_k=[];cache_v=[]
 for i in range(8):
  k=np.zeros((8,smax,64),np.float32);v=np.zeros_like(k);k[:,:initial,:]=preout[1+i*2].reshape(8,initial,64);v[:,:initial,:]=preout[2+i*2].reshape(8,initial,64);cache_k.append(k);cache_v.append(v)
 inv=np.load(d/'rotary_inv_freq.npy').astype('float32');token=int(np.asarray(preout[0]).argmax());tokens=[token];kvtime=[];blocktime=[];headtime=[];writetime=[]
 for step in range(z.max_new):
  if step and token in eos:break
  x=np.asarray(table[token:token+1],dtype=np.float32).reshape(1,1,512);freq=np.concatenate((inv*(initial+step),inv*(initial+step)))[None,None,:];co=np.cos(freq).astype('float32');si=np.sin(freq).astype('float32');mask=np.full((8,1,smax),-10000.,np.float32);mask[:,:,:initial+step+1]=0
  for i in range(8):
   t=time.perf_counter();dk,dv=kvs[i].inference(inputs=[x,co,si]);kvtime.append(time.perf_counter()-t);t=time.perf_counter();cache_k[i][:,initial+step:initial+step+1,:]=dk;cache_v[i][:,initial+step:initial+step+1,:]=dv;writetime.append(time.perf_counter()-t);t=time.perf_counter();x=blocks[i].inference(inputs=[x,cache_k[i],cache_v[i],co,si,mask])[0];blocktime.append(time.perf_counter()-t)
  pieces=[]
  for h in heads:t=time.perf_counter();pieces.append(h.inference(inputs=[x])[0]);headtime.append(time.perf_counter()-t)
  token=int(np.concatenate(pieces,axis=-1).argmax());tokens.append(token)
 out={'complete_to_eos':tokens[-1] in eos,'eos_ids':sorted(eos),'generated_tokens':len(tokens)-1,'tokens':tokens,'latency_ms':{'encoder':encoder_s*1000,'adapter':adapter_s*1000,'prefill':prefill_s*1000,'decode_npu_per_token':float((sum(kvtime)+sum(blocktime)+sum(headtime))/max(len(kvtime)//8,1)*1000),'cpu_kv_write_per_token':float(sum(writetime)/max(len(writetime)//8,1)*1000),'end_to_end':(time.perf_counter()-start)*1000},'placement':'all neural compute uses RKNN NPU; CPU does embedding lookup, RoPE/mask, KV writes and argmax','rss_mib':{'before':before,'after':rss()}};z.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
finally:
 for r in allr:r.release()
