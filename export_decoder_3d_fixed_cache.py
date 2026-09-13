"""Export the complete Qwen2 token decoder with fixed-size rank-3 CPU KV buffers.

Each layer has two RKNN graphs: projection (new K/V) and read-only decoder
block.  Cache lives exclusively in host-owned [H, Smax, D] arrays; an input
mask excludes the unused tail, so static RKNN graph shapes support all steps.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, onnx, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor

H, D, C = 8, 64, 512
PROMPT = 'Please transcribe this audio.'
def rot(x): return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)
class KV(torch.nn.Module):
 def __init__(self,l): super().__init__();self.n=l.input_layernorm;self.k=l.self_attn.k_proj;self.v=l.self_attn.v_proj
 def forward(self,x,c,s):
  x=self.n(x); k=self.k(x).reshape(H,1,D);v=self.v(x).reshape(H,1,D);c=c.reshape(1,1,D);s=s.reshape(1,1,D);return k*c+rot(k)*s,v
class Block(torch.nn.Module):
 def __init__(self,l):
  super().__init__();self.n=l.input_layernorm;self.q=l.self_attn.q_proj;self.o=l.self_attn.o_proj;self.p=l.post_attention_layernorm;self.g=l.mlp.gate_proj;self.u=l.mlp.up_proj;self.d=l.mlp.down_proj;self.scale=l.self_attn.scaling
 def forward(self,x,k,v,c,s,mask):
  q=self.q(self.n(x)).reshape(H,1,D);c=c.reshape(1,1,D);s=s.reshape(1,1,D);q=q*c+rot(q)*s
  prob=torch.softmax((q@k.transpose(-2,-1))*self.scale+mask,dim=-1);attn=(prob@v).reshape(1,1,C);r=x+self.o(attn);z=self.p(r);return r+self.d(F.silu(self.g(z))*self.u(z))
class Head(torch.nn.Module):
 def __init__(self,m):super().__init__();self.n=m.language_model.model.norm;self.h=m.language_model.lm_head
 def forward(self,x):return self.h(self.n(x))
def save(p,x):np.save(p,x.detach().cpu().float().numpy())
def main():
 a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--audio',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--smax',type=int,default=128);args=a.parse_args()
 torch.set_grad_enabled(False); p=AutoProcessor.from_pretrained(args.model,trust_remote_code=True);m=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
 conv=[{'role':'user','content':[{'type':'audio','path':str(args.audio)},{'type':'text','text':PROMPT}]}]
 b=dict(p.apply_chat_template(conv,return_tensors='pt',sampling_rate=16000,audio_padding='max_length',add_generation_prompt=True,audio_max_length=128000,text_kwargs={'padding':'longest','truncation':True,'max_length':1000}))
 with torch.inference_mode():
  e=m._inject_audio_embeddings(b['input_ids'],b['input_features'],b.get('feature_lens'));pre=m.language_model(inputs_embeds=e,use_cache=True,return_dict=True);tok=pre.logits[:,-1,:].argmax(-1,keepdim=True); x=m.get_input_embeddings()(tok);seq=pre.past_key_values.layers[0].keys.shape[-2];assert seq+1<=args.smax;pos=torch.tensor([[seq]]);co,si=m.language_model.model.rotary_emb(x,pos); mask=torch.full((H,1,args.smax),-10000.);mask[:,:,:seq+1]=0; manifest={'prefill_length':int(seq),'smax':args.smax,'first_token':int(tok),'layers':[]}
  for idx,l in enumerate(m.language_model.model.layers):
   out=args.output/f'layer{idx}';out.mkdir(parents=True,exist_ok=True);kv=KV(l).eval();bl=Block(l).eval();dk,dv=kv(x,co,si);old=pre.past_key_values.layers[idx];fk=torch.zeros(H,args.smax,D);fv=torch.zeros_like(fk);fk[:,:seq,:]=old.keys.reshape(H,seq,D);fv[:,:seq,:]=old.values.reshape(H,seq,D);fk[:,seq:seq+1,:]=dk;fv[:,seq:seq+1,:]=dv;y=bl(x,fk,fv,co,si,mask)
   for n,v in {'hidden':x,'cos':co,'sin':si,'full_k':fk,'full_v':fv,'mask':mask,'key_delta_reference':dk,'value_delta_reference':dv,'hidden_reference':y}.items():save(out/(n+'.npy'),v)
   torch.onnx.export(kv,(x,co,si),out/'kv.onnx',input_names=['hidden','cos','sin'],output_names=['key_delta','value_delta'],opset_version=19,dynamo=False)
   torch.onnx.export(bl,(x,fk,fv,co,si,mask),out/'block.onnx',input_names=['hidden','full_k','full_v','cos','sin','mask'],output_names=['hidden_out'],opset_version=19,dynamo=False)
   onnx.checker.check_model(onnx.load(out/'kv.onnx'));onnx.checker.check_model(onnx.load(out/'block.onnx'));manifest['layers'].append(idx);x=y
  hd=Head(m).eval(); logits=hd(x); ho=args.output/'head';ho.mkdir(parents=True,exist_ok=True);save(ho/'hidden',x);save(ho/'logits_reference',logits);torch.onnx.export(hd,x,ho/'head.onnx',input_names=['hidden'],output_names=['logits'],opset_version=19,dynamo=False);onnx.checker.check_model(onnx.load(ho/'head.onnx'));manifest['reference_next_token']=int(logits.argmax(-1))
 args.output.mkdir(parents=True,exist_ok=True);(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
