import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
p=argparse.ArgumentParser();p.add_argument('--dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bshd',action='store_true');p.add_argument('--allnchw',action='store_true');a=p.parse_args();d=a.dir
h=np.load(d/'hidden.npy').astype('float32');cos=np.load(d/'cos.npy').astype('float32');sin=np.load(d/'sin.npy').astype('float32');mask=np.load(d/'mask.npy').astype('float32');fk=np.load(d/'full_k.npy').astype('float32');fv=np.load(d/'full_v.npy').astype('float32');kr=np.load(d/'key_delta_reference.npy').astype('float32');vr=np.load(d/'value_delta_reference.npy').astype('float32');ref=np.load(d/'hidden_reference.npy').astype('float32');kv=RKNNLite(verbose=False);bl=RKNNLite(verbose=False)
try:
 assert kv.load_rknn(str(d/'kv_fp16.rknn'))==0 and kv.init_runtime()==0;block_name='block_allnchw_fp16.rknn' if a.allnchw else ('block_bshd_fp16.rknn' if a.bshd else 'block_fp16.rknn');assert bl.load_rknn(str(d/block_name))==0 and bl.init_runtime()==0;dk,dv=kv.inference(inputs=[h,cos,sin]);cpu_concat_start=time.perf_counter();fk[:,:,-1:,:]=dk;fv[:,:,-1:,:]=dv;concat=(time.perf_counter()-cpu_concat_start)*1e3;bk,bv=(fk.transpose(0,2,1,3).copy(),fv.transpose(0,2,1,3).copy()) if a.bshd else (fk,fv);payload=[h,bk,bv,cos,sin,mask];formats=None
 if a.allnchw: payload=[h.reshape(1,1,1,512),bk,bv,cos.reshape(1,1,1,64),sin.reshape(1,1,1,64),mask];formats='nchw'
 bl.inference(inputs=payload,data_format=formats);ts=[]
 for _ in range(20):t=time.perf_counter();out=bl.inference(inputs=payload,data_format=formats)[0];ts.append(time.perf_counter()-t)
 def err(x,y):return float(np.linalg.norm(x.astype('float64')-y.astype('float64'))/np.linalg.norm(y.astype('float64')))
 z={'kv_l2':[err(dk,kr),err(dv,vr)],'block_hidden_l2':err(out,ref),'cache_append_ms':concat,'block_latency_ms':float(np.mean(ts)*1e3)};a.output.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
finally:kv.release();bl.release()
