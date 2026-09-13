"""Split Qwen2 final norm + LM head into NPU-sized vocabulary shards."""
import argparse
from pathlib import Path
import numpy as np, onnx, torch
from transformers import AutoModelForCausalLM
class Shard(torch.nn.Module):
 def __init__(self,norm,weight,bias):
  super().__init__();self.norm=norm;self.register_buffer('weight',weight);self.register_buffer('bias',bias)
 def forward(self,x):return torch.nn.functional.linear(self.norm(x),self.weight,self.bias)
a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--hidden',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--shards',type=int,default=8);z=a.parse_args()
m=AutoModelForCausalLM.from_pretrained(z.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='eager').eval();x=torch.from_numpy(np.load(z.hidden)).float();head=m.language_model.lm_head;v=head.weight.shape[0];z.output.mkdir(parents=True,exist_ok=True)
for i,(lo,hi) in enumerate(zip(np.linspace(0,v,z.shards+1,dtype=int)[:-1],np.linspace(0,v,z.shards+1,dtype=int)[1:])):
 s=Shard(m.language_model.model.norm,head.weight[lo:hi],None if head.bias is None else head.bias[lo:hi]).eval();d=z.output/f'shard{i:02d}';d.mkdir(exist_ok=True);ref=s(x);np.save(d/'logits_reference.npy',ref.detach().numpy());torch.onnx.export(s,x,d/'head.onnx',input_names=['hidden'],output_names=['logits'],opset_version=19,dynamo=False);onnx.checker.check_model(onnx.load(d/'head.onnx'));(d/'range.txt').write_text(f'{lo} {hi}\n');print(i,lo,hi)
