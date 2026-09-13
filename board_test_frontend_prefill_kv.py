"""Audio frontend RKNN -> real prefill RKNN K/V bridge for one fixed prompt."""
import argparse,json,time
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite
a=argparse.ArgumentParser();a.add_argument('--dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);z=a.parse_args();d=z.dir
paths=[('/root/audio8-asr/encoder_f800_v2/audio_encoder_f800_fp16.rknn','enc'),('/root/audio8-asr/audio_adapter_h104_t100/audio_adapter_h104_t100_fp16.rknn','adp'),(str(d/'prefill_kv_s110_fp16.rknn'),'pre')];rs=[]
try:
 for p,_ in paths:r=RKNNLite(verbose=False);assert r.load_rknn(p)==0 and r.init_runtime()==0;rs.append(r)
 x=np.load(d/'input_features.npy').astype('float32');template=np.load(d/'input_embeddings.npy').astype('float32');positions=np.load(d/'audio_positions.npy').reshape(-1);t=time.perf_counter();e=rs[0].inference(inputs=[x])[0];te=time.perf_counter()-t;t=time.perf_counter();audio=rs[1].inference(inputs=[e])[0];ta=time.perf_counter()-t;assert len(positions)==len(audio),(positions.shape,audio.shape);template[0,positions,:]=audio;t=time.perf_counter();o=rs[2].inference(inputs=[template]);tp=time.perf_counter()-t;out={'audio_positions':len(positions),'frontend_shapes':{'encoder':list(e.shape),'adapter':list(audio.shape),'prefill_input':list(template.shape)},'latency_ms':{'encoder':te*1000,'adapter':ta*1000,'prefill':tp*1000},'first_token':int(o[0].argmax()),'kv_outputs':len(o)-1,'kv_shapes':[list(x.shape) for x in o[1:]]};z.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
finally:
 for r in rs:r.release()
