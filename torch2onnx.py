import torch
import torch.nn as nn

import sys
sys.path.append('./src/KHI/network/pytorch/EXAM')

from src.KHI.network.pytorch.EXAM.carvinet import MultiNetV10_BB as carvinet

class NET(nn.Module):
	def __init__(self, num_class=7) -> None:
		super().__init__()
		
		self.act = nn.ReLU()

		# 48,16
		# 24, 8
		# 12, 4
		# 6, 2
		# 3, 1
		self.conv_model = torch.nn.Sequential(

			torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
			torch.nn.ReLU(),

			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32),
			torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
			torch.nn.ReLU(),

			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True, groups=64),
			torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
			torch.nn.ReLU(),

			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32),
			torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True),
			torch.nn.ReLU(),

			torch.nn.Conv2d(in_channels=16, out_channels=7, kernel_size=1, stride=1, padding=0, bias=True),
		)


		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True)
		
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32)
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)

		self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True, groups=32)
		self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)

		self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32)
		self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)

		self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True, groups=32)
		self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)

		self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32)
		self.conv11 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)

		self.conv12 = torch.nn.Conv2d(in_channels=16, out_channels=7, kernel_size=1, stride=1, padding=0, bias=True)

		#self.fc1 = torch.nn.Linear(32*4*4, 7)
		#self.softmax = torch.log_softmax(dim=1)

	def forward(self, x):

		#x = self.conv_model(x)
		x = self.act(self.conv1(x))

		x = self.conv2(x)
		x = self.act(self.conv3(x))
		add1 = x

		x = self.conv4(x)
		x = self.act(self.conv5(x))
		x = add1 + x

		x = self.conv6(x)
		x = self.act(self.conv7(x))
		add2 = x

		x = self.conv8(x)
		x = self.act(self.conv9(x))
		x = add2 + x

		x = self.conv10(x)
		x = self.act(self.conv11(x))

		x = self.conv12(x)
		x = torch.flatten(x, 1)
		return x

import numpy as np
if __name__=='__main__':

	#model = NET(7)

	model = carvinet(7)

	x = torch.randn(1, 3, 320, 320, requires_grad=True)
	dummy_data = torch.empty(1, 3, 320, 320, dtype=torch.float32)
	torch_out = model(x)

	import torch.onnx
	
	torch.onnx.export(model, dummy_data, './src/KHI/utils/torch2onnx.onnx',
					export_params=True,
					do_constant_folding=True,
					verbose=True,
					opset_version=10)


	import onnx
	from onnx import numpy_helper

	onnx_model = onnx.load('./src/KHI/utils/torch2onnx.onnx')
	onnx.checker.check_model(onnx_model)
	import struct
	struct.unpack("f", b"&\276!\300")

	weight=[]

	for init in onnx_model.graph.initializer:
		w = numpy_helper.to_array(init)
		weight.append(w)

	for node in onnx_model.graph.node:
		temp_input = []
		for i in node.input:
			#if i.isdigit():
			temp_input.append(i)
		print(f'inputs: {temp_input}, outputs: {node.output}, op_type: {node.op_type}, name: {node.name}')
		
		#print(node.input)
		#print(node.attribute)
		#for at in node.attribute:
		#	print(at)

	import onnxruntime

	ort_session = onnxruntime.InferenceSession('./src/KHI/utils/torch2onnx.onnx')

	def to_numpy(tensor):
		return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

	# ONNX 런타임에서 계산된 결과값
	ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
	ort_outs = ort_session.run(None, ort_inputs)

	# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
	np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

	print("Exported model has been tested with ONNXRuntime, and the result looks good!")
