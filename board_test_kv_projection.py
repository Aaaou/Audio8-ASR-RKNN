import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
p=argparse.ArgumentParser();p.add_argument('--dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=a.dir;x=np.load(d/'hidden.npy').astype('float32');refs=[np.load(d/n).astype('float32') for n in ('key_reference.npy','value_reference.npy')];r=RKNNLite(verbose=False)
try:
 assert r.load_rknn(str(d/'kv_projection_fp16.rknn'))==0 and r.init_runtime()==0;r.inference(inputs=[x]);ts=[]
 for _ in range(20):t=time.perf_counter();out=r.inference(inputs=[x]);ts.append(time.perf_counter()-t)
 rows=[]
 for n,v,ref in zip(('key_delta','value_delta'),out,refs):rows.append({'name':n,'l2':float(np.linalg.norm(v-ref)/np.linalg.norm(ref)),'max_abs':float(np.abs(v-ref).max())})
 z={'latency_ms':float(np.mean(ts)*1e3),'outputs':rows};a.output.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
finally:r.release()
