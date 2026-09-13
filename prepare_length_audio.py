"""Create duration-controlled WAV inputs for end-to-end ASR validation."""
import argparse
from pathlib import Path
import soundfile as sf
a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--seconds',type=float,nargs='+',required=True);z=a.parse_args();x,sr=sf.read(z.input);z.output_dir.mkdir(parents=True,exist_ok=True)
for sec in z.seconds:
 p=z.output_dir/f's{sec:g}.wav';sf.write(p,x[:round(sec*sr)],sr);print(p,sec,sr)
