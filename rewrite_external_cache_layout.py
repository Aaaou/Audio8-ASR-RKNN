"""Make cache inputs BS-H-D externally, then explicitly transpose to BH-S-D."""
import argparse, onnx
from onnx import helper
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();m=onnx.load(a.input);g=m.graph
for old in ('full_k','full_v'):
 vi=next(x for x in g.input if x.name==old); dims=vi.type.tensor_type.shape.dim; dims[1].dim_value,dims[2].dim_value=dims[2].dim_value,dims[1].dim_value
 new=old+'_bhsd'; node=helper.make_node('Transpose',[old],[new],name=old+'_external_to_attention',perm=[0,2,1,3]);g.node.insert(0,node)
 for n in g.node[1:]: n.input[:]=[new if x==old else x for x in n.input]
onnx.checker.check_model(m);onnx.save(m,a.output)
