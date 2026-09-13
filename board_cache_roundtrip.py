import numpy as np,json
from rknnlite.api import RKNNLite
k=np.load('full_k.npy').astype('float32');v=np.load('full_v.npy').astype('float32');r=RKNNLite(verbose=False)
try:
 assert r.load_rknn('cache_roundtrip.rknn')==0 and r.init_runtime()==0;o=r.inference(inputs=[k,v],inputs_pass_through=[1,1]); print(RKNNLite.inference.__doc__)
 candidates={'same':k,'bshd_to_bhsd_reshape':k.transpose(0,2,1,3).reshape(k.shape),'bhsd_to_bshd_reshape':k.reshape(1,111,8,64).transpose(0,2,1,3)}
 print(json.dumps({'k_l2':float(np.linalg.norm(o[0]-k)/np.linalg.norm(k)),'mapping':{n:float(np.linalg.norm(o[0]-x)/np.linalg.norm(x)) for n,x in candidates.items()},'shapes':[list(x.shape) for x in o]},indent=2))
finally:r.release()
