import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
p=argparse.ArgumentParser();p.add_argument('--dir',type=Path,required=True);p.add_argument('--runs',type=int,default=5);p.add_argument('--output',type=Path,required=True);p.add_argument('--transpose-cache',action='store_true');a=p.parse_args();d=a.dir;m=json.loads((d/'manifest.json').read_text());ins=[np.load(d/(n+'.npy')).astype('float32') for n in m['inputs']];
if a.transpose_cache: ins=[x.transpose(0,2,1,3).copy() if i else x for i,x in enumerate(ins)]
r=RKNNLite(verbose=False)
try:
 assert r.load_rknn(str(d/'layer0_debug_fp16.rknn'))==0 and r.init_runtime()==0;r.inference(inputs=ins);ts=[]
 for _ in range(a.runs):
  t=time.perf_counter();out=r.inference(inputs=ins);ts.append(time.perf_counter()-t)
 names=json.loads((d/'debug_outputs.json').read_text());rows=[]
 for n,x in zip(names,out):
  ref=np.load(d/(n.replace('/','_')+'_reference.npy')).astype('float32');delta=x.astype('float64')-ref.astype('float64');rows.append({'name':n,'shape':list(x.shape),'l2':float(np.linalg.norm(delta)/np.linalg.norm(ref)),'max_abs':float(np.abs(delta).max())})
 payload={'latency_ms':{'mean':float(np.mean(ts)*1e3),'p50':float(np.percentile(ts,50)*1e3)},'outputs':rows};a.output.write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))
finally:r.release()
