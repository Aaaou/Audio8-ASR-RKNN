"""Export current-token K/V projection only; cache ownership stays on CPU."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch,onnx
from transformers import AutoModelForCausalLM,AutoProcessor
PROMPT='Please transcribe this audio.'
class KV(torch.nn.Module):
 def __init__(self,a): super().__init__(); self.k=a.k_proj; self.v=a.v_proj
 def forward(self,h):
  shape=(1,1,8,64)
  return self.k(h).view(shape).transpose(1,2),self.v(h).view(shape).transpose(1,2)
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--audio',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_grad_enabled(False);pr=AutoProcessor.from_pretrained(a.model,trust_remote_code=True);m=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='eager').eval();c=[{'role':'user','content':[{'type':'audio','path':str(a.audio)},{'type':'text','text':PROMPT}]}];b=dict(pr.apply_chat_template(c,return_tensors='pt',sampling_rate=16000,audio_padding='max_length',add_generation_prompt=True,audio_max_length=128000,text_kwargs={'padding':'longest','truncation':True,'max_length':1000}))
with torch.inference_mode():
 emb=m._inject_audio_embeddings(b['input_ids'],b['input_features'],b.get('feature_lens'));o=m.language_model(inputs_embeds=emb,use_cache=True,return_dict=True);tok=o.logits[:,-1,:].argmax(-1,keepdim=True);h=m.language_model.model.layers[0].input_layernorm(m.get_input_embeddings()(tok));w=KV(m.language_model.model.layers[0].self_attn).eval();k,v=w(h)
out=a.output;out.mkdir(parents=True,exist_ok=True);np.save(out/'hidden.npy',h.numpy());np.save(out/'key_reference.npy',k.numpy());np.save(out/'value_reference.npy',v.numpy());torch.onnx.export(w,(h,),out/'kv_projection.onnx',input_names=['hidden'],output_names=['key_delta','value_delta'],opset_version=19,dynamo=False);onnx.checker.check_model(onnx.load(out/'kv_projection.onnx'));(out/'manifest.json').write_text(json.dumps({'onnx':str(out/'kv_projection.onnx'),'input':'hidden.npy','outputs':['key_delta','value_delta'],'references':['key_reference.npy','value_reference.npy'],'token':int(tok)}));print((out/'manifest.json').read_text())
