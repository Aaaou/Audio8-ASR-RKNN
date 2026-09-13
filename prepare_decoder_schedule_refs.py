"""Generate greedy CPU references used to strictly validate RKNN KV scheduling."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoProcessor
P='Please transcribe this audio.'
a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--audio',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--steps',type=int,default=6);z=a.parse_args();torch.set_grad_enabled(False)
p=AutoProcessor.from_pretrained(z.model,trust_remote_code=True);m=AutoModelForCausalLM.from_pretrained(z.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='eager').eval();c=[{'role':'user','content':[{'type':'audio','path':str(z.audio)},{'type':'text','text':P}]}];b=dict(p.apply_chat_template(c,return_tensors='pt',sampling_rate=16000,audio_padding='max_length',add_generation_prompt=True,audio_max_length=128000,text_kwargs={'padding':'longest','truncation':True,'max_length':1000}))
captured=[]
def hook(_, __, output): captured.append((output[0] if isinstance(output, tuple) else output).detach().clone())
hooks=[layer.register_forward_hook(hook) for layer in m.language_model.model.layers]
with torch.inference_mode():
 e=m._inject_audio_embeddings(b['input_ids'],b['input_features'],b.get('feature_lens'));o=m.language_model(inputs_embeds=e,use_cache=True,return_dict=True);past=o.past_key_values;token=o.logits[:,-1,:].argmax(-1,keepdim=True);seq=int(past.layers[0].keys.shape[-2]);meta={'initial_length':seq,'input_tokens':[],'output_tokens':[]}
 for step in range(z.steps):
  x=m.get_input_embeddings()(token);pos=torch.tensor([[seq+step]]);co,si=m.language_model.model.rotary_emb(x,pos);d=z.output/f'step{step}';d.mkdir(parents=True,exist_ok=True)
  np.save(d/'hidden.npy',x.numpy());np.save(d/'cos.npy',co.numpy());np.save(d/'sin.npy',si.numpy());meta['input_tokens'].append(int(token))
  captured.clear();o=m.language_model(inputs_embeds=x,past_key_values=past,use_cache=True,cache_position=torch.tensor([seq+step]),position_ids=pos,output_hidden_states=True,return_dict=True);past=o.past_key_values;assert len(captured)==8
  for i in range(8):
   np.save(d/f'layer{i}_hidden.npy',captured[i].numpy());np.save(d/f'layer{i}_key.npy',past.layers[i].keys[:,:,seq+step:seq+step+1,:].reshape(8,1,64).numpy());np.save(d/f'layer{i}_value.npy',past.layers[i].values[:,:,seq+step:seq+step+1,:].reshape(8,1,64).numpy())
  token=o.logits[:,-1,:].argmax(-1,keepdim=True);meta['output_tokens'].append(int(token))
for h in hooks:h.remove()
z.output.mkdir(parents=True,exist_ok=True);(z.output/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
