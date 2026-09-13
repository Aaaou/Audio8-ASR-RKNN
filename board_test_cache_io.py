import numpy as np,json
from pathlib import Path
from rknnlite.api import RKNNLite
d=Path('.');h=np.load('hidden.npy').astype('float32');k=np.load('full_k.npy').astype('float32');v=np.load('full_v.npy').astype('float32');c=np.load('cos.npy').astype('float32');s=np.load('sin.npy').astype('float32');m=np.load('mask.npy').astype('float32');r=RKNNLite(verbose=False)
try:
 assert r.load_rknn('block_cacheio_fp16.rknn')==0 and r.init_runtime()==0;o=r.inference(inputs=[h,k,v,c,s,m]);
 def l(a,b):return float(np.linalg.norm(a.astype('float64')-b.astype('float64'))/np.linalg.norm(b.astype('float64')))
 print(json.dumps({'hidden_l2':l(o[0],np.load('hidden_reference.npy')),'returned_key_l2':l(o[1],k),'returned_value_l2':l(o[2],v),'shapes':[list(x.shape) for x in o]},indent=2))
finally:r.release()
