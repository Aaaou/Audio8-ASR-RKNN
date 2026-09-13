import torch,onnx
class I(torch.nn.Module):
 def forward(self,k,v): return k+torch.zeros_like(k),v+torch.zeros_like(v)
x=torch.randn(1,8,111,64);m=I().eval();torch.onnx.export(m,(x,x),'cache_roundtrip.onnx',input_names=['k','v'],output_names=['ko','vo'],opset_version=19,dynamo=False);onnx.checker.check_model(onnx.load('cache_roundtrip.onnx'))
