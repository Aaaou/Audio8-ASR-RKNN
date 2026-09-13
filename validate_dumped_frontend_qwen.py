"""Compare CPU-original versus dumped NPU frontend embeddings and first logits."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM,AutoProcessor
PROMPT="Please transcribe this audio."
def metric(a,r):
 a,r=a.astype(np.float64),r.astype(np.float64);d=a-r
 return {"normalized_l2":float(np.linalg.norm(d)/np.linalg.norm(r)),"max_abs":float(abs(d).max()),"cosine":float((a*r).sum()/(np.linalg.norm(a)*np.linalg.norm(r))),"argmax_match":bool(a.argmax()==r.argmax()),"actual_argmax":int(a.argmax()),"reference_argmax":int(r.argmax())}
p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--dump",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True);model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation="eager").eval();d=np.load(args.dump,allow_pickle=True);rows=[]
for row in json.loads(args.manifest.read_text()):
 if row["id"] not in d:continue
 c=[{"role":"user","content":[{"type":"audio","path":row["audio"]},{"type":"text","text":PROMPT}]}];b=dict(processor.apply_chat_template(c,return_tensors="pt",sampling_rate=16000,audio_padding="max_length",add_generation_prompt=True,audio_max_length=128000,text_kwargs={"padding":"longest","truncation":True,"max_length":1000}))
 x=d[row["id"]].item(); ids=torch.from_numpy(x["input_ids"]);pos=torch.nonzero(ids[0].eq(model.audio_token_id),as_tuple=False).flatten()
 with torch.inference_mode():
  cpu=model._inject_audio_embeddings(b["input_ids"],b["input_features"],b.get("feature_lens"));cpu_logits=model.language_model(inputs_embeds=cpu,use_cache=False,return_dict=True).logits[:,-1,:]
 npu_embed=model.get_input_embeddings()(ids).clone();npu_embed[0,pos,:]=torch.from_numpy(x["npu_audio_embeddings"]);npu_logits=model.language_model(inputs_embeds=npu_embed,use_cache=False,return_dict=True).logits[:,-1,:]
 rows.append({"id":row["id"],"embedding":metric(npu_embed[0,pos,:].detach().numpy(),cpu[0,pos,:].detach().numpy()),"cpu_qwen_logits":metric(npu_logits.detach().numpy(),cpu_logits.detach().numpy())})
args.output.write_text(json.dumps({"scope":"CPU Qwen2 prefill fed by CPU or frozen FP16 NPU frontend embeddings","samples":rows},indent=2));print(args.output.read_text())
