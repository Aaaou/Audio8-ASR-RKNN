"""Dump NPU frontend embeddings/IDs without loading PyTorch model weights."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from transformers import AutoProcessor
from rknnlite.api import RKNNLite
PROMPT="Please transcribe this audio."
p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,required=True); p.add_argument("--encoder-rknn",type=Path,required=True); p.add_argument("--adapter-rknn",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--limit",type=int,default=2); args=p.parse_args()
audio_token_id=json.loads((args.model/"config.json").read_text())["audio_token_id"]
processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True); es=[RKNNLite(verbose=False),RKNNLite(verbose=False)]
try:
 for e,path in zip(es,(args.encoder_rknn,args.adapter_rknn)):
  if e.load_rknn(str(path))!=0 or e.init_runtime()!=0: raise RuntimeError("RKNN init failed")
 output={}
 for row in json.loads(args.manifest.read_text())[:args.limit]:
  conversation=[{"role":"user","content":[{"type":"audio","path":row["audio"]},{"type":"text","text":PROMPT}]}]
  b=dict(processor.apply_chat_template(conversation,return_tensors="pt",sampling_rate=16000,audio_padding="max_length",add_generation_prompt=True,audio_max_length=128000,text_kwargs={"padding":"longest","truncation":True,"max_length":1000}))
  raw=np.asarray(es[0].inference(inputs=[b["input_features"][0].float().numpy().astype(np.float32)])[0],np.float32)
  projected=np.asarray(es[1].inference(inputs=[raw])[0],np.float32)
  output[row["id"]]={"input_ids":b["input_ids"].numpy(),"npu_audio_embeddings":projected,"npu_encoder_hidden":raw}
 np.savez(args.output,**output)
 print(json.dumps({"output":str(args.output),"ids":list(output),"prompt_shapes":{k:list(v["input_ids"].shape) for k,v in output.items()}},indent=2))
finally:
 for e in es:e.release()
