"""Expose selected layer-0 intermediates from the static decode ONNX graph."""
import argparse, onnx
from onnx import helper
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();g=onnx.load(a.input);g=onnx.shape_inference.infer_shapes(g);wanted=['/language_model/model/layers.0/self_attn/Concat_2_output_0','/language_model/model/layers.0/self_attn/Concat_3_output_0','/language_model/model/layers.0/self_attn/Softmax_output_0','/language_model/model/layers.0/self_attn/MatMul_1_output_0','/language_model/model/layers.0/Add_output_0','/language_model/model/layers.0/Add_1_output_0']
known={x.name:x for x in list(g.graph.value_info)+list(g.graph.output)}
for name in wanted:
 if name in known and not any(x.name==name for x in g.graph.output): g.graph.output.append(known[name])
onnx.checker.check_model(g);onnx.save(g,a.output);print('exposed',[(x.name, x.type.tensor_type.shape) for x in g.graph.output if x.name in wanted])
