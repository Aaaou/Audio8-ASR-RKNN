import argparse,json
from pathlib import Path
import numpy as np,onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--dir',type=Path,required=True);a=p.parse_args();m=json.loads((a.dir/'manifest.json').read_text());inputs={n:np.load(a.dir/(n+'.npy')).astype('float32') for n in m['inputs']};o=ort.InferenceSession(str(a.dir/'layer0_debug.onnx'),providers=['CPUExecutionProvider']).run(None,inputs)
names=[x.name for x in ort.InferenceSession(str(a.dir/'layer0_debug.onnx'),providers=['CPUExecutionProvider']).get_outputs()]
for n,v in zip(names,o):np.save(a.dir/(n.replace('/','_')+'_reference.npy'),v)
(a.dir/'debug_outputs.json').write_text(json.dumps(names,indent=2));print(names)
