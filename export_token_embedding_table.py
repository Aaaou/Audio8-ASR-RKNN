"""Export Qwen token embedding table used by the CPU-side decode scheduler."""
import argparse
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM
a=argparse.ArgumentParser();a.add_argument('--model',type=Path,required=True);a.add_argument('--output',type=Path,required=True);z=a.parse_args()
m=AutoModelForCausalLM.from_pretrained(z.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='eager').eval();x=m.get_input_embeddings().weight.detach().cpu().float().numpy();z.output.parent.mkdir(parents=True,exist_ok=True);np.save(z.output,x);print(x.shape,x.dtype,x.nbytes)
