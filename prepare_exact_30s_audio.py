"""Create a deterministic exactly-30-second 16 kHz mono WAV for official-cap validation."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
x,sr=sf.read(a.input,dtype='float32',always_2d=False)
if x.ndim==2:x=x.mean(axis=1)
if sr!=16000: raise ValueError(f'expected source at 16 kHz, got {sr}')
n=30*16000
x=np.pad(x[:n],(0,max(0,n-len(x))))
a.output.parent.mkdir(parents=True,exist_ok=True)
sf.write(a.output,x,sr)
print({'path':str(a.output),'samples':len(x),'seconds':len(x)/sr})
