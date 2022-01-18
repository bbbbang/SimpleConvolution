import torch

import numpy as np



def custom_depthwise_conv(input_data, weight, bias, height, width, padding, stride, kernel, in_channel, out_channel):
	output_height = (height + 2*padding[1] - kernel[1]) // stride[1] + 1
	output_width = (width + 2*padding[0] - kernel[0]) // stride[0] + 1
	output_size = (out_channel, output_height,  output_width)
	output_data = np.zeros(output_size).flatten()

	pad_width = width + padding[1]*2
	pad_height = height + padding[0]*2
	pad_input = np.zeros((in_channel, height + padding[0]*2, width + padding[1]*2)).flatten()

	if padding == (1,1):
		for i in range(0, in_channel):
			for r in range(0, pad_height):
				for c in range(0, pad_width):
					if r == 0 or c == 0 or r == pad_height - 1 or c == pad_width-1:
						pad_input[i * pad_height * pad_width + r * pad_width + c] = 0
					else:
						pad_input[i * pad_height * pad_width + r * pad_width + c] = input_data[i * height * width + (r-1) * width + (c-1)]

	output_pos = 0
	kernels = kernel[0]*kernel[0]

	for in_ch in range(0, in_channel):
		for row in range(0, pad_height-kernel[0]+1, stride[1]):
			for col in range(0, pad_width-kernel[0]+1, stride[0]):
				output_pos = in_ch * output_height * output_width + row//stride[1] * output_width + col//stride[0]

				output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 0) * pad_width + (col + 0)] * weight[in_ch * kernels + 0]
				output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 0) * pad_width + (col + 1)] * weight[in_ch * kernels + 1]
				output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 0) * pad_width + (col + 2)] * weight[in_ch * kernels + 2]
				output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 1) * pad_width + (col + 0)] * weight[in_ch * kernels + 3]
				output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 1) * pad_width + (col + 1)] * weight[in_ch * kernels + 4]
				output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 1) * pad_width + (col + 2)] * weight[in_ch * kernels + 5]
				output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 2) * pad_width + (col + 0)] * weight[in_ch * kernels + 6]
				output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 2) * pad_width + (col + 1)] * weight[in_ch * kernels + 7]
				output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 2) * pad_width + (col + 2)] * weight[in_ch * kernels + 8]

				output_data[output_pos] += bias[in_ch]

		#for row in range(0, output_height):
		#	for col in range(0, output_width):
		#		output_data[in_ch * output_height * output_width + row * output_width + col] += bias[in_ch]

	return output_data


def custom_pointwise_conv(input_data, weight, bias, height, width, padding, stride, kernel, in_channel, out_channel):

	output_height = height
	output_width = width
	output_size = (out_channel, output_height,  output_width)
	output_data = np.zeros(output_size).flatten()

	output_pos = 0

	for out_ch in range(0, out_channel):
		for in_ch in range(0, in_channel):
			for row in range(0, height):
				for col in range(0, width):
					output_pos = out_ch * output_height * output_width + row * output_width + col
					output_data[output_pos] += input_data[in_ch * height * width + row * width + col] * weight[out_ch * in_channel + in_ch]

		for row in range(0, output_height):
			for col in range(0, output_width):
				output_data[out_ch * output_height * output_width + row * output_width + col] += bias[out_ch]

	return output_data


def custom_conv(input_data, weight, bias, height, width, padding, stride, kernel, in_channel, out_channel):

	output_height = (height + 2*padding[1] - kernel[1]) // stride[1] + 1
	output_width = (width + 2*padding[0] - kernel[0]) // stride[0] + 1
	output_size = (out_channel, output_height,  output_width)
	output_data = np.zeros(output_size).flatten()

	pad_width = width + padding[1]*2
	pad_height = height + padding[0]*2
	pad_input = np.zeros((in_channel, height + padding[0]*2, width + padding[1]*2)).flatten()

	if padding == (1,1):
		for i in range(0, in_channel):
			for r in range(0, pad_height):
				for c in range(0, pad_width):
					if r == 0 or c == 0 or r == pad_height - 1 or c == pad_width-1:
						pad_input[i * pad_height * pad_width + r * pad_width + c] = 0
					else:
						pad_input[i * pad_height * pad_width + r * pad_width + c] = input_data[i * height * width + (r-1) * width + (c-1)]

	output_pos = 0
	kernels = kernel[0]*kernel[0]

	for out_ch in range(0, out_channel):
		for in_ch in range(0, in_channel):
			for row in range(0, pad_height-kernel[0]+1, stride[1]):
				for col in range(0, pad_width-kernel[0]+1, stride[0]):
					output_pos = out_ch * output_height * output_width + row//stride[1] * output_width + col//stride[0]

					output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 0) * pad_width + (col + 0)] * weight[out_ch * in_channel * kernels + in_ch * kernels + 0]
					output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 0) * pad_width + (col + 1)] * weight[out_ch * in_channel * kernels + in_ch * kernels + 1]
					output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 0) * pad_width + (col + 2)] * weight[out_ch * in_channel * kernels + in_ch * kernels + 2]
					output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 1) * pad_width + (col + 0)] * weight[out_ch * in_channel * kernels + in_ch * kernels + 3]
					output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 1) * pad_width + (col + 1)] * weight[out_ch * in_channel * kernels + in_ch * kernels + 4]
					output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 1) * pad_width + (col + 2)] * weight[out_ch * in_channel * kernels + in_ch * kernels + 5]
					output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 2) * pad_width + (col + 0)] * weight[out_ch * in_channel * kernels + in_ch * kernels + 6]
					output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 2) * pad_width + (col + 1)] * weight[out_ch * in_channel * kernels + in_ch * kernels + 7]
					output_data[output_pos] += pad_input[in_ch * pad_height * pad_width + (row + 2) * pad_width + (col + 2)] * weight[out_ch * in_channel * kernels + in_ch * kernels + 8]

		for row in range(0, output_height):
			for col in range(0, output_width):
				output_data[out_ch * output_height * output_width + row * output_width + col] += bias[out_ch]

	return output_data


def custom_conv_1(input_data, weight, bias, height, width, padding, stride, kernel, in_channel, out_channel):

	output_size = (out_channel, (height + 2*padding[1] - kernel[1]) // stride[1] + 1,  (width + 2*padding[0] - kernel[0]) // stride[0] + 1)
	output_data = np.zeros(output_size)

	pad_width = width + padding[1]*2
	pad_height = height + padding[0]*2
	pad_input = np.zeros((in_channel, height + padding[0]*2, width + padding[1]*2))

	if padding == (1,1):
		for i in range(0, in_channel):
			for r in range(0, pad_height):
				for c in range(0, pad_width):
					if r == 0 or c == 0 or r == pad_height - 1 or c == pad_width-1:
						pad_input[i,r,c] = 0
					else:
						pad_input[i,r,c] = input_data[0,i,r-1,c-1]

	for out_ch in range(0, out_channel):
		for in_ch in range(0, in_channel):
			for row in range(0, pad_height-kernel[0]+1, stride[1]):
				for col in range(0, pad_width-kernel[0]+1, stride[0]):
					output_data[out_ch, row//stride[1], col//stride[1]] += pad_input[in_ch, row+0, col+0] * weight[out_ch, in_ch, 0, 0]
					output_data[out_ch, row//stride[1], col//stride[1]] += pad_input[in_ch, row+0, col+1] * weight[out_ch, in_ch, 0, 1]
					output_data[out_ch, row//stride[1], col//stride[1]] += pad_input[in_ch, row+0, col+2] * weight[out_ch, in_ch, 0, 2]
					output_data[out_ch, row//stride[1], col//stride[1]] += pad_input[in_ch, row+1, col+0] * weight[out_ch, in_ch, 1, 0]
					output_data[out_ch, row//stride[1], col//stride[1]] += pad_input[in_ch, row+1, col+1] * weight[out_ch, in_ch, 1, 1]
					output_data[out_ch, row//stride[1], col//stride[1]] += pad_input[in_ch, row+1, col+2] * weight[out_ch, in_ch, 1, 2]
					output_data[out_ch, row//stride[1], col//stride[1]] += pad_input[in_ch, row+2, col+0] * weight[out_ch, in_ch, 2, 0]
					output_data[out_ch, row//stride[1], col//stride[1]] += pad_input[in_ch, row+2, col+1] * weight[out_ch, in_ch, 2, 1]
					output_data[out_ch, row//stride[1], col//stride[1]] += pad_input[in_ch, row+2, col+2] * weight[out_ch, in_ch, 2, 2]

		for row in range(0, pad_height-kernel[0]+1, stride[1]):
			for col in range(0, pad_width-kernel[0]+1, stride[0]):
				output_data[out_ch, row//stride[1], col//stride[1]] += bias[out_ch]

	return output_data



def make_numpy(item):
	return item.detach().cpu().numpy().flatten()



if __name__=='__main__':


	test_input = torch.ones((1,3,16,16))
	test_input_numpy = test_input.detach().cpu().numpy()
	test_input_numpy_flatten = test_input_numpy.flatten()

	normal_conv_k3s2p1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True)
	normal_conv_k3s1p1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
	
	pointwise_conv_k1s1p0 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)

	depthwise_conv_k3s2p1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1, bias=True, groups=3)
	depthwise_conv_k3s1p1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, groups=3)


	# pytorch output
	torch_output_normal_conv_k3s2p1 = normal_conv_k3s2p1(test_input)
	torch_output_normal_conv_k3s1p1 = normal_conv_k3s1p1(test_input)

	torch_output_pointwise_conv_k1s1p0 = pointwise_conv_k1s1p0(test_input)

	torch_output_depthwise_conv_k3s2p1 = depthwise_conv_k3s2p1(test_input)
	torch_output_depthwise_conv_k3s1p1 = depthwise_conv_k3s1p1(test_input)


	# custom output
	numpy_test_input = make_numpy(test_input)

	custom_output_normal_conv_k3s2p1 = custom_conv(numpy_test_input, make_numpy(normal_conv_k3s2p1.weight), make_numpy(normal_conv_k3s2p1.bias), 16, 16, (1,1), (2,2), (3,3), 3, 32)
	custom_output_normal_conv_k3s1p1 = custom_conv(numpy_test_input, make_numpy(normal_conv_k3s1p1.weight), make_numpy(normal_conv_k3s1p1.bias), 16, 16, (1,1), (1,1), (3,3), 3, 32)

	custom_output_pointwise_conv_k1s1p0 = custom_pointwise_conv(numpy_test_input, make_numpy(pointwise_conv_k1s1p0.weight), make_numpy(pointwise_conv_k1s1p0.bias), 16, 16, (0,0), (1,1), (1,1), 3, 32)

	custom_output_depthwise_conv_k3s2p1 = custom_depthwise_conv(numpy_test_input, make_numpy(depthwise_conv_k3s2p1.weight), make_numpy(depthwise_conv_k3s2p1.bias), 16, 16, (1,1), (2,2), (3,3), 3, 3)
	custom_output_depthwise_conv_k3s1p1 = custom_depthwise_conv(numpy_test_input, make_numpy(depthwise_conv_k3s1p1.weight), make_numpy(depthwise_conv_k3s1p1.bias), 16, 16, (1,1), (1,1), (3,3), 3, 3)


	error_normal_conv_k3s2p1 = (abs(make_numpy(torch_output_normal_conv_k3s2p1) - custom_output_normal_conv_k3s2p1)).mean()
	error_normal_conv_k3s1p1 = (abs(make_numpy(torch_output_normal_conv_k3s1p1) - custom_output_normal_conv_k3s1p1)).mean()

	error_pointwise_conv_k1s1p0 = (abs(make_numpy(torch_output_pointwise_conv_k1s1p0) - custom_output_pointwise_conv_k1s1p0)).mean()

	error_depthwise_conv_k3s2p1 = (abs(make_numpy(torch_output_depthwise_conv_k3s2p1) - custom_output_depthwise_conv_k3s2p1)).mean()
	error_depthwise_conv_k3s1p1 = (abs(make_numpy(torch_output_depthwise_conv_k3s1p1) - custom_output_depthwise_conv_k3s1p1)).mean()


	print(error_normal_conv_k3s2p1)
	print(error_normal_conv_k3s1p1)
	print(error_pointwise_conv_k1s1p0)
	print(error_depthwise_conv_k3s2p1)
	print(error_depthwise_conv_k3s1p1)
