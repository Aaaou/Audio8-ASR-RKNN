"""Wrap rank-3 decoder inputs as rank-4 so RKNN Lite can use NCHW for all."""
import argparse,onnx
from onnx import helper,numpy_helper
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();m=onnx.load(a.input);g=m.graph;prefix=[]
for name,shape in [('hidden',[1,1,512]),('cos',[1,1,64]),('sin',[1,1,64])]:
 vi=next(x for x in g.input if x.name==name);vi.name=name+'_nchw4';dims=vi.type.tensor_type.shape.dim;dims.insert(2,onnx.TensorShapeProto.Dimension(dim_value=1));shape_name=name+'_shape';g.initializer.append(numpy_helper.from_array(np.asarray(shape,np.int64),shape_name));prefix.append(helper.make_node('Reshape',[vi.name,shape_name],[name],name=name+'_restore_rank3'))
for node in reversed(prefix): g.node.insert(0,node)
onnx.checker.check_model(m);onnx.save(m,a.output)
