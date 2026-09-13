import argparse,onnx
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();m=onnx.load(a.input);g=m.graph;known={x.name:x for x in list(g.input)+list(g.value_info)+list(g.output)}
for n in ('full_k','full_v'):
 if not any(x.name==n for x in g.output):g.output.append(known[n])
onnx.checker.check_model(m);onnx.save(m,a.output)
