"""Export one fixed-cache Qwen2 decode step with cache-delta outputs.

Inputs are one token embedding and 16 historical K/V tensors at sequence
length S. Outputs are next-token logits plus only the new [1,8,1,64] K/V
entries for every layer.  Keeping concatenation outside the first RKNN graph
avoids relying on mutable/cache-reorder behavior before numerical recurrence is
proven.  This script generates the S=110 CPU reference from a real Audio8
prefill and exports the step for RKNN conversion.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import onnx, torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.cache_utils import DynamicCache

PROMPT="Please transcribe this audio."

class DecodeStepDelta(torch.nn.Module):
    def __init__(self, language_model: torch.nn.Module, layers: int):
        super().__init__(); self.language_model=language_model; self.layers=layers
    def forward(self, token_embedding: torch.Tensor, *past: torch.Tensor):
        cache=DynamicCache(ddp_cache_data=tuple((past[2*i],past[2*i+1]) for i in range(self.layers)))
        out=self.language_model(inputs_embeds=token_embedding,past_key_values=cache,use_cache=True,return_dict=True)
        values=[out.logits[:,-1,:]]
        for layer in out.past_key_values.layers:
            values += [layer.keys[:,:,-1:,:],layer.values[:,:,-1:,:]
            ]
        return tuple(values)

def main():
 p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--audio",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--max-audio-seconds",type=int,default=8);a=p.parse_args()
 torch.set_grad_enabled(False);processor=AutoProcessor.from_pretrained(a.model,trust_remote_code=True);model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation="eager").eval()
 c=[{"role":"user","content":[{"type":"audio","path":str(a.audio)},{"type":"text","text":PROMPT}]}];b=dict(processor.apply_chat_template(c,return_tensors="pt",sampling_rate=16000,audio_padding="max_length",add_generation_prompt=True,audio_max_length=a.max_audio_seconds*16000,text_kwargs={"padding":"longest","truncation":True,"max_length":1000}))
 with torch.inference_mode():
  emb=model._inject_audio_embeddings(b["input_ids"],b["input_features"],b.get("feature_lens"));pre=model.language_model(inputs_embeds=emb,use_cache=True,return_dict=True)
  token=pre.logits[:,-1,:].argmax(-1,keepdim=True);token_embedding=model.get_input_embeddings()(token)
  past=tuple(x for layer in pre.past_key_values.layers for x in (layer.keys,layer.values));wrapper=DecodeStepDelta(model.language_model,len(pre.past_key_values.layers)).eval();reference=wrapper(token_embedding,*past)
 out=a.output;out.mkdir(parents=True,exist_ok=True);np.save(out/"token_embedding.npy",token_embedding.numpy())
 names=["token_embedding"]
 for i,x in enumerate(past): names.append(f"past_{i//2}_{'key' if i%2==0 else 'value'}");np.save(out/(names[-1]+".npy"),x.numpy())
 output_names=["last_logits"]
 for i in range(len(past)//2):output_names += [f"delta_{i}_key",f"delta_{i}_value"]
 for n,x in zip(output_names,reference):np.save(out/(n+"_reference.npy"),x.numpy())
 onnx_path=out/f"qwen2_decode_step_s{past[0].shape[-2]}_delta.onnx"
 torch.onnx.export(wrapper,(token_embedding,*past),onnx_path,input_names=names,output_names=output_names,opset_version=19,do_constant_folding=True,dynamo=False)
 onnx.checker.check_model(onnx.load(onnx_path))
 manifest={"onnx":str(onnx_path),"cache_length":int(past[0].shape[-2]),"layers":len(past)//2,"token":int(token.item()),"next_token":int(reference[0].argmax().item()),"inputs":names,"outputs":output_names}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
