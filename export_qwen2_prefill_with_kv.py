"""Export fixed-length Qwen2 prefill logits plus all initial K/V tensors."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,onnx,torch
from transformers import AutoModelForCausalLM,AutoProcessor
P='Please transcribe this audio.'
class PrefillKV(torch.nn.Module):
 def __init__(self,lm):super().__init__();self.lm=lm
 def forward(self,x):
  o=self.lm(inputs_embeds=x,use_cache=True,return_dict=True)
  y=[o.logits[:,-1,:]]
  for layer in o.past_key_values.layers:y.extend([layer.keys,layer.values])
  return tuple(y)
a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--audio',type=Path,required=True);a.add_argument('--output',type=Path,required=True);z=a.parse_args();torch.set_grad_enabled(False)
p=AutoProcessor.from_pretrained(z.model,trust_remote_code=True);m=AutoModelForCausalLM.from_pretrained(z.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='eager').eval();c=[{'role':'user','content':[{'type':'audio','path':str(z.audio)},{'type':'text','text':P}]}];b=dict(p.apply_chat_template(c,return_tensors='pt',sampling_rate=16000,audio_padding='max_length',add_generation_prompt=True,audio_max_length=128000,text_kwargs={'padding':'longest','truncation':True,'max_length':1000}))
with torch.inference_mode():x=m._inject_audio_embeddings(b['input_ids'],b['input_features'],b.get('feature_lens'));w=PrefillKV(m.language_model).eval();ref=w(x)
z.output.mkdir(parents=True,exist_ok=True);np.save(z.output/'input_embeddings.npy',x.numpy());np.save(z.output/'input_features.npy',b['input_features'][0].float().numpy());np.save(z.output/'input_ids.npy',b['input_ids'].cpu().to(torch.int64).numpy());np.save(z.output/'audio_positions.npy',(b['input_ids'][0]==m.config.audio_token_id).nonzero().cpu().to(torch.int64).numpy());np.save(z.output/'rotary_inv_freq.npy',m.language_model.model.rotary_emb.inv_freq.cpu().float().numpy());names=['last_logits']+[f'layer{i}_{t}' for i in range(8) for t in ('key','value')]
for n,v in zip(names,ref):np.save(z.output/(n+'.npy'),v.numpy())
onnx_path=z.output/f'prefill_kv_s{x.shape[1]}.onnx';torch.onnx.export(w,x,onnx_path,input_names=['inputs_embeds'],output_names=names,opset_version=19,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path));manifest={'sequence':x.shape[1],'outputs':names,'argmax':int(ref[0].argmax()),'kv_shape':list(ref[1].shape)};(z.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
