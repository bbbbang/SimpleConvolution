import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from dataloader_tl import Dataloader_TL as dataloader


class Conv_BN(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, bias=False):
		super().__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride, padding=padding, groups=groups, bias=False)
		self.bn = nn.BatchNorm2d(num_features=out_channels)
	def forward(self, x):
		if hasattr(self, 'fused_conv'):
			return self.fused_conv(x)
		return self.bn(self.conv(x))


class BlazeBlock(nn.Module):
	def __init__(self, in_channels, out_channels_1, out_channels_2=None, kernel_size=5, stride=1, use_double=False):
		super().__init__()\

		padding = kernel_size // 2

		self.use_pooling = False
		self.use_double = use_double

		if self.use_double:
			self.channel_pad = out_channels_2 - in_channels
		else:
			self.channel_pad = out_channels_1 - in_channels
		
		if stride == 2:
			self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
			self.use_pooling = True

		self.act = nn.ReLU()

		self.dwconv1 = Conv_BN(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
		self.pwconv1 = Conv_BN(in_channels=in_channels, out_channels=out_channels_1, kernel_size=1, stride=1, padding=0, groups=1)

		if self.use_double:
			self.dwconv2 = Conv_BN(in_channels=out_channels_1, out_channels=out_channels_1, kernel_size=kernel_size, stride=1, padding=padding, groups=out_channels_1)
			self.pwconv2 = Conv_BN(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=1, stride=1, padding=0, groups=1)

	def forward(self, x):
		x1 = self.pwconv1(self.dwconv1(x))

		if self.use_double:
			x1 = self.pwconv2(self.dwconv2(x1))

		if self.use_pooling:
			x = self.pool(x)
		if self.channel_pad > 0:
			x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)

		x1 = self.act(x1 + x)
		return x1


class Backbone(nn.Module):
	def __init__(self):
		super().__init__()

		self.act = nn.ReLU()

		self.conv1 = Conv_BN(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1, groups=1)

		self.sbb01 = BlazeBlock(in_channels=24, out_channels_1=24, out_channels_2=None, kernel_size=3, stride=1, use_double=False) # 56
		self.sbb02 = BlazeBlock(in_channels=24, out_channels_1=24, out_channels_2=None, kernel_size=3, stride=1, use_double=False) # 56 
		self.sbb03 = BlazeBlock(in_channels=24, out_channels_1=48, out_channels_2=None, kernel_size=3, stride=2, use_double=False) # 28
		self.sbb04 = BlazeBlock(in_channels=48, out_channels_1=48, out_channels_2=None, kernel_size=3, stride=1, use_double=False) # 28
		self.sbb05 = BlazeBlock(in_channels=48, out_channels_1=48, out_channels_2=None, kernel_size=3, stride=1, use_double=False) # 28
		self.dbb01 = BlazeBlock(in_channels=48, out_channels_1=24, out_channels_2=96, 	kernel_size=3, stride=2, use_double=True) # 14
		self.dbb02 = BlazeBlock(in_channels=96, out_channels_1=24, out_channels_2=96, 	kernel_size=3, stride=1, use_double=True) # 14
		self.dbb03 = BlazeBlock(in_channels=96, out_channels_1=24, out_channels_2=96, 	kernel_size=3, stride=1, use_double=True) # 14
		self.dbb04 = BlazeBlock(in_channels=96, out_channels_1=24, out_channels_2=96, 	kernel_size=3, stride=2, use_double=True) # 7
		self.dbb05 = BlazeBlock(in_channels=96, out_channels_1=24, out_channels_2=96, 	kernel_size=3, stride=1, use_double=True) # 7
		self.dbb06 = BlazeBlock(in_channels=96, out_channels_1=24, out_channels_2=96, 	kernel_size=3, stride=1, use_double=True) # 7

	def forward(self, x):
		x = self.act(self.conv1(x))
		x = self.sbb01(x)
		x = self.sbb02(x)
		x = self.sbb03(x); p2 = x
		x = self.sbb04(x)
		x = self.sbb05(x)
		x = self.dbb01(x); p3 = x
		x = self.dbb02(x)
		x = self.dbb03(x)
		x = self.dbb04(x); p4 = x
		x = self.dbb05(x)
		x = self.dbb06(x); p5 = x
		return p2, p3, p4, p5


class Neck(nn.Module):
	def __init__(self, in_28x28, in_14x14, in_7_7, in_backbone, out_28x28, out_14x14, out_7x7):
		super().__init__()
		self.act = nn.ReLU()
		self.up = nn.UpsamplingNearest2d(scale_factor=2)

		self.conv_28x28 = Conv_BN(in_channels=in_28x28, out_channels=out_28x28, kernel_size=1, stride=1, padding=0, groups=1)

		self.conv_14x14 = Conv_BN(in_channels=in_14x14, out_channels=out_14x14, kernel_size=1, stride=1, padding=0, groups=1)

		self.conv_7x7 = Conv_BN(in_channels=in_7_7, out_channels=out_7x7, kernel_size=1, stride=1, padding=0, groups=1)

		self.conv_backbone = Conv_BN(in_channels=in_backbone, out_channels=out_7x7, kernel_size=1, stride=1, padding=0, groups=1)

		self.dwconv_7x7 = Conv_BN(in_channels=out_7x7, out_channels=out_7x7, kernel_size=3, stride=1, padding=1, groups=out_7x7)
		self.pwconv_7x7 = Conv_BN(in_channels=out_7x7, out_channels=out_7x7, kernel_size=1, stride=1, padding=0, groups=1)

		self.dwconv_14x14 = Conv_BN(in_channels=out_14x14, out_channels=out_14x14, kernel_size=3, stride=1, padding=1, groups=out_14x14)
		self.pwconv_14x14 = Conv_BN(in_channels=out_14x14, out_channels=out_14x14, kernel_size=1, stride=1, padding=0, groups=1)

		self.dwconv_28x28 = Conv_BN(in_channels=out_28x28, out_channels=out_28x28, kernel_size=3, stride=1, padding=1, groups=out_28x28)
		self.pwconv_28x28 = Conv_BN(in_channels=out_28x28, out_channels=out_28x28, kernel_size=1, stride=1, padding=0, groups=1)


	def forward(self, x_28x28, x_14x14, x_7x7, x_backbone):
		x_28x28 = self.conv_28x28(x_28x28)
		x_14x14 = self.conv_14x14(x_14x14)
		x_7x7 = self.conv_7x7(x_7x7)

		x = self.conv_backbone(x_backbone)
		x = self.act(x + x_7x7)
		x = self.pwconv_7x7(self.dwconv_7x7(x))
		x = self.up(x)
		x = self.act(x + x_14x14)
		x = self.pwconv_14x14(self.dwconv_14x14(x))
		x = self.up(x)
		x = self.act(x + x_28x28)
		x = self.pwconv_28x28(self.dwconv_28x28(x))

		return x


class Head(nn.Module):
	def __init__(self, in_channels, out_channels, num_class=2):
		super().__init__()
		self.act = nn.ReLU()

		self.offset_dwconv1 = Conv_BN(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
		self.offset_pwconv1 = Conv_BN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
		self.offset_dwconv2 = Conv_BN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
		self.offset_pwconv2 = Conv_BN(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
		self.offset_out = nn.Conv2d(in_channels=out_channels, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)

		self.size_dwconv1 = Conv_BN(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
		self.size_pwconv1 = Conv_BN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
		self.size_dwconv2 = Conv_BN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
		self.size_pwconv2 = Conv_BN(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
		self.size_out = nn.Conv2d(in_channels=out_channels, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)

		self.keypoint_dwconv1 = Conv_BN(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
		self.keypoint_pwconv1 = Conv_BN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
		self.keypoint_dwconv2 = Conv_BN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
		self.keypoint_pwconv2 = Conv_BN(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
		self.keypoint_out = nn.Conv2d(in_channels=out_channels, out_channels=num_class, kernel_size=1, stride=1, padding=0, bias=True)

	def forward(self, x):
		
		x_offset = self.act(self.offset_pwconv1(self.offset_dwconv1(x)))
		x_offset = self.act(self.offset_pwconv2(self.offset_dwconv2(x_offset)))
		x_offset = self.offset_out(x_offset)

		x_size = self.act(self.size_pwconv1(self.size_dwconv1(x)))
		x_size = self.act(self.size_pwconv2(self.size_dwconv2(x_size)))
		x_size = self.size_out(x_size)

		x_keypoint = self.act(self.keypoint_pwconv1(self.keypoint_dwconv1(x)))
		x_keypoint = self.act(self.keypoint_pwconv2(self.keypoint_dwconv2(x_keypoint)))
		x_keypoint = torch.sigmoid(self.keypoint_out(x_keypoint))

		return (x_offset, x_size, x_keypoint)
		#outputs = {'reg': x_offset,'wh': x_size,'cls': x_keypoint}
		#return outputs


class TLNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.backbone = Backbone()
		self.neck = Neck(in_28x28=48, in_14x14=96, in_7_7=96, in_backbone=96, out_28x28=40, out_14x14=40, out_7x7=40)
		self.head = Head(in_channels=40, out_channels=40, num_class=2)

	def forward(self, x):
		x_28x28, x_14x14, x_7x7, x_backbone = self.backbone(x)
		x = self.neck(x_28x28, x_14x14, x_7x7, x_backbone)
		x = self.head(x)
		return {'reg': x[0],'wh': x[1],'cls': x[2]}




def build_trainer():

	trainer = dict()

	return trainer


def train(_trainer):

	print('start training')




def fuse_bn(kernel, bn):
	gamma = bn.weight
	std = (bn.running_var + bn.eps).sqrt()
	return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

def get_modules_blank(module):
	items = module._modules.items()
	for _, item in items:
		if len(item._modules) > 0:
			get_modules_blank(item)
		if isinstance(item, Conv_BN) and hasattr(item, 'conv'):

			print(type(item))

			kernel, bias = fuse_bn(item.conv.weight, item.bn)

			in_channels = item.conv.in_channels
			out_channels = item.conv.out_channels
			kernel_size = item.conv.kernel_size
			stride = item.conv.stride
			padding = item.conv.padding
			groups = item.conv.groups

			new_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
			new_conv.weight.data = kernel
			new_conv.bias.data = bias

			item.add_module('fused_conv', new_conv)

			delattr(item, 'conv')
			delattr(item, 'bn')

			for para in item.parameters():
				para.detach_()


import struct

def export_weight_binary(model, output_path):

	def get_layers(layers):
		items = layers._modules.items()
		for _, item in items:
			if len(item._modules) > 0:
				get_layers(item)
			if hasattr(item, 'weight'):
				weight_shape = item.weight.shape
				bias_shape = item.bias.shape
				groups = item.groups
				kernel_size = item.kernel_size
				padding = item.padding
				stride = item.stride
				in_channel = item.in_channels
				out_channel = item.out_channels

				conv_type = None
				if in_channel == groups:
					conv_type = 'depthwise'
				elif groups == 1 and kernel_size == (1,1):
					conv_type = 'pointwise'
				else:
					conv_type = 'conv'

				print(f'{conv_type} : w({weight_shape}), b({bias_shape}), in({in_channel}), out({out_channel}), k({kernel_size}), s({stride}), p({padding}), g({groups})')

				flatten_weight = torch.flatten(item.weight)
				flatten_weight = flatten_weight.detach().cpu().numpy().tolist()

				flatten_bias = torch.flatten(item.bias)
				flatten_bias = flatten_bias.detach().cpu().numpy().tolist()

				if in_channel == 3:
					print(flatten_weight)
					print(flatten_bias)

				bin_data = list()
				bin_data.append(9000)
				bin_data.append(float(in_channel))
				bin_data.append(float(out_channel))
				bin_data.append(float(kernel_size[0]))
				bin_data.append(float(stride[0]))
				bin_data.append(float(padding[0]))
				bin_data.append(float(groups))
				bin_data.append(flatten_weight)
				bin_data.append(flatten_bias)

				for data in bin_data:
					if isinstance(data, list):
						for w in data:
							#if w < 1e-6:
							#	w = 0
							bin_weight = struct.pack('f', w)
							bin_file.write(bin_weight)
						continue
					c = struct.pack('f', data)
					bin_file.write(c)

	bin_file = open(output_path, 'wb')

	get_layers(model)

	bin_file.close()


def get_feature(self, input, output):
	print(output)

if __name__=='__main__':

	model = TLNet()

	def initialize(module):
		if isinstance(module, nn.BatchNorm2d):
			nn.init.normal_(module.weight.data)
			nn.init.normal_(module.bias.data)

	#model.apply(initialize)

	import torchsummary
	model = model.eval()
	torchsummary.summary(model, (3, 112, 112), 1, device='cpu')
	

	print('before')
	get_modules_blank(model)

	print('after')
	get_modules_blank(model)


	model.backbone.conv1.register_forward_hook(get_feature)

	temp = torch.ones((1, 3,160,160))
	model(temp)


	export_weight_binary(model, './detection_test.w')

	# flatten_weight_file = './src/KHI/utils/detection1'
	# bin_file = open(flatten_weight_file + '.w', 'wb')
	# export_binary(model, bin_file)
	# bin_file.close()



	import torch.onnx
	dummy_data = torch.empty(1, 3, 112, 112, dtype=torch.float32)
	torch.onnx.export(model, dummy_data, './src/KHI/utils/detection_model.onnx',
					export_params=True,
					do_constant_folding=True,
					verbose=True,
					opset_version=10)

	trainer = build_trainer()
	train(trainer)
