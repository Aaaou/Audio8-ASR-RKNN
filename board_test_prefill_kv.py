"""Verify RKNN prefill's logits and all 8 initial K/V outputs on RK3576."""
import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
def l2(a,b):return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/np.linalg.norm(b.astype(np.float64)))
a=argparse.ArgumentParser();a.add_argument('--dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);z=a.parse_args();d=z.dir;x=np.load(d/'input_embeddings.npy').astype('float32');r=RKNNLite(verbose=False)
try:
 assert r.load_rknn(str(d/'prefill_kv_s110_fp16.rknn'))==0 and r.init_runtime()==0;r.inference(inputs=[x]);t=time.perf_counter();o=r.inference(inputs=[x]);elapsed=(time.perf_counter()-t)*1000;names=['last_logits']+[f'layer{i}_{q}' for i in range(8) for q in ('key','value')];errs={n:l2(v,np.load(d/(n+'.npy'))) for n,v in zip(names,o)};ref=np.load(d/'last_logits.npy');out={'latency_ms':elapsed,'logits_l2':errs['last_logits'],'argmax':[int(o[0].argmax()),int(ref.argmax())],'max_kv_l2':max(v for n,v in errs.items() if n!='last_logits'),'outputs':errs};z.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
finally:r.release()
