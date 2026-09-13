"""Export all eight Qwen2 decode layers without graph-internal cache updates.

For each layer the KV graph produces a RoPE-applied delta.  The block graph
consumes a fixed, already CPU-appended full cache and never uses DynamicCache,
Concat, ScatterND, or mutable cache state.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import onnx, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT = "Please transcribe this audio."

def rotate(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

class KV(torch.nn.Module):
    def __init__(self, layer):
        super().__init__(); self.norm=layer.input_layernorm; self.k=layer.self_attn.k_proj; self.v=layer.self_attn.v_proj
    def forward(self, hidden, cos, sin):
        x=self.norm(hidden); k=self.k(x).view(1,1,8,64).transpose(1,2); v=self.v(x).view(1,1,8,64).transpose(1,2)
        c=cos.unsqueeze(1); s=sin.unsqueeze(1)
        return k*c+rotate(k)*s, v

class Block(torch.nn.Module):
    def __init__(self, layer):
        super().__init__(); self.norm=layer.input_layernorm; self.q=layer.self_attn.q_proj; self.o=layer.self_attn.o_proj; self.post=layer.post_attention_layernorm; self.gate=layer.mlp.gate_proj; self.up=layer.mlp.up_proj; self.down=layer.mlp.down_proj; self.scale=layer.self_attn.scaling
    def forward(self, hidden, full_k, full_v, cos, sin, mask):
        q=self.q(self.norm(hidden)).view(1,1,8,64).transpose(1,2); c=cos.unsqueeze(1); s=sin.unsqueeze(1); q=q*c+rotate(q)*s
        score=torch.matmul(q,full_k.transpose(-2,-1))*self.scale+mask
        attn=torch.matmul(torch.softmax(score.float(),dim=-1).to(q.dtype),full_v).transpose(1,2).reshape(1,1,512)
        residual=hidden+self.o(attn); x=self.post(residual); return residual+self.down(F.silu(self.gate(x))*self.up(x))

def save(path, tensor): np.save(path, tensor.detach().cpu().float().numpy())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",type=Path,required=True);ap.add_argument("--audio",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    torch.set_grad_enabled(False); p=AutoProcessor.from_pretrained(args.model,trust_remote_code=True);m=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation="eager").eval()
    convo=[{"role":"user","content":[{"type":"audio","path":str(args.audio)},{"type":"text","text":PROMPT}]}]
    b=dict(p.apply_chat_template(convo,return_tensors="pt",sampling_rate=16000,audio_padding="max_length",add_generation_prompt=True,audio_max_length=128000,text_kwargs={"padding":"longest","truncation":True,"max_length":1000}))
    with torch.inference_mode():
        e=m._inject_audio_embeddings(b["input_ids"],b["input_features"],b.get("feature_lens")); pre=m.language_model(inputs_embeds=e,use_cache=True,return_dict=True); token=pre.logits[:,-1,:].argmax(-1,keepdim=True); hidden=m.get_input_embeddings()(token); pos=torch.tensor([[110]]); cos,sin=m.language_model.model.rotary_emb(hidden,pos)
        mask=torch.zeros(1,1,1,111)
        meta=[]
        for i, layer in enumerate(m.language_model.model.layers):
            out=args.output/f"layer{i}";out.mkdir(parents=True,exist_ok=True); kv=KV(layer).eval(); block=Block(layer).eval(); old=pre.past_key_values.layers[i]; dk,dv=kv(hidden,cos,sin); fk=torch.cat((old.keys,dk),-2);fv=torch.cat((old.values,dv),-2);next_hidden=block(hidden,fk,fv,cos,sin,mask)
            values={"hidden":hidden,"cos":cos,"sin":sin,"full_k":fk,"full_v":fv,"mask":mask,"key_delta_reference":dk,"value_delta_reference":dv,"hidden_reference":next_hidden}
            for name,value in values.items(): save(out/(name+".npy"),value)
            torch.onnx.export(kv,(hidden,cos,sin),out/"kv.onnx",input_names=["hidden","cos","sin"],output_names=["key_delta","value_delta"],opset_version=19,dynamo=False)
            torch.onnx.export(block,(hidden,fk,fv,cos,sin,mask),out/"block.onnx",input_names=["hidden","full_k","full_v","cos","sin","mask"],output_names=["hidden_out"],opset_version=19,dynamo=False)
            onnx.checker.check_model(onnx.load(out/"kv.onnx"));onnx.checker.check_model(onnx.load(out/"block.onnx"));meta.append({"layer":i,"dir":str(out),"cache_length":111})
            hidden=next_hidden
    (args.output/"manifest.json").write_text(json.dumps({"token":int(token),"layers":meta},indent=2));print((args.output/"manifest.json").read_text())
if __name__=="__main__": main()
