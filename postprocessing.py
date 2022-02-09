import torch
import torch.nn as nn
import numpy as np
from src.KHI.network.pytorch.nn_torch import *
#from torchsummary import summary
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time



class Postprocessing_landmark_torch(Module):
	def __init__(self, class_num=7, input_size=320):
		super().__init__()
		self.class_num = class_num
		self.tensor_dim = input_size / 4.

	def gather_nd_torch(self, params, indices):
		gathered = params[indices[:,1], indices[:,0]]
		return gathered

	def tr(self, tensor):
		m = tensor.transpose(1,2)
		m = m.transpose(2,3)
		return m

	def forward(self, x):
		top_k = 10

		offset, size, keypoint, landmark, landmark_offset = x
		
		landmark_offset = torch.squeeze(self.tr(landmark_offset), dim=0)

		landmark_fmap_max = F.max_pool2d(landmark, 3, stride=1, padding=1)
		landmark_keep = (torch.abs(landmark_fmap_max - landmark) < 1e-6).float()
		landmark_b = landmark * landmark_keep

		landmark_b = torch.squeeze(self.tr(landmark_b), dim=0)
		landmark_c = torch.flatten(landmark_b)
		landmark_conf, landmark_index = torch.topk(landmark_c, top_k)

		landmark_channel_indices_temp = landmark_index // 4
		landmark_y_indices = landmark_channel_indices_temp // self.tensor_dim
		landmark_x_indices = landmark_channel_indices_temp - landmark_y_indices * self.tensor_dim
		landmark_channel_indices = landmark_index - landmark_channel_indices_temp * 4

		landmark_classes = landmark_channel_indices

		landmark_points = torch.stack([landmark_x_indices, landmark_y_indices], dim=1)
		landmark_points = landmark_points.long()
		landmark_offsets = self.gather_nd_torch(landmark_offset, landmark_points)
		landmark_points = landmark_points + landmark_offsets
		landmark_points = landmark_points * 4.

		offset = torch.squeeze(self.tr(offset), dim=0)
		size = torch.squeeze(self.tr(size), dim=0)

		fmap_max = F.max_pool2d(keypoint, 3, stride=1, padding=1)
		keep = (torch.abs(fmap_max - keypoint) < 1e-6).float()
		b = keypoint * keep

		b = torch.squeeze(self.tr(b), dim=0)
		c = torch.flatten(b)
		detection_scores, index = torch.topk(c, top_k)

		channel_indices_temp = index // self.class_num
		y_indices = channel_indices_temp // self.tensor_dim
		x_indices = channel_indices_temp - (y_indices * self.tensor_dim)
		detection_classes = index - (channel_indices_temp * self.class_num)

		combined_indices = torch.stack([y_indices, x_indices], dim=-1)

		y_indices = y_indices.long()
		x_indices = x_indices.long()
		sizes = size[y_indices, x_indices]
		offsets = offset[y_indices, x_indices]

		sizes = torch.flip(sizes, [1])
		offsets = torch.flip(offsets, [1])

		pos = combined_indices + offsets

		zero_tensors = torch.zeros((top_k, self.class_num))
		cur_device = sizes.device
		zero_tensors = zero_tensors.to(cur_device)

		hight_width = torch.maximum(sizes, zero_tensors)
		hight_width = hight_width / 2

		min_pos = pos - hight_width
		max_pos = pos + hight_width

		boxes = torch.cat([min_pos, max_pos], dim=1)
		boxes = torch.clip(boxes, 0., self.tensor_dim-1.)
		boxes = boxes * 4.

		return boxes, detection_classes, detection_scores, landmark_points, landmark_classes, landmark_conf

class Postprocessing_torch(Module):
	def __init__(self, class_num=7, input_size=320):
		super().__init__()
		self.class_num = class_num
		self.tensor_dim = input_size / 4.

	def tr(self, tensor):
		m = tensor.transpose(1,2)
		m = m.transpose(2,3)
		return m

	def forward(self, x):
		top_k = 10
		offset, size, keypoint = x

		offset = torch.permute(offset, (0,2,3,1))
		size = torch.permute(size, (0,2,3,1))
		offset = torch.squeeze(offset, dim=0)
		size = torch.squeeze(size, dim=0)

		fmap_max = F.max_pool2d(keypoint, 3, stride=1, padding=1)
		keep = torch.eq(fmap_max, keypoint).float()
		#keep = (torch.abs(fmap_max - keypoint) < 1e-6).float()
		b = keypoint * keep

		b = torch.permute(b, (0,2,3,1))
		b = torch.squeeze(b, dim=0)
		c = torch.flatten(b)
		detection_scores, index = torch.topk(c, top_k)

		channel_indices_temp = index // self.class_num
		#channel_indices_temp = index
		y_indices = channel_indices_temp // self.tensor_dim
		x_indices = channel_indices_temp - (y_indices * self.tensor_dim)
		detection_classes = index - (channel_indices_temp * self.class_num)
		#detection_classes = channel_indices_temp
		combined_indices = torch.stack([y_indices, x_indices], dim=-1)

		y_indices = y_indices.long()
		x_indices = x_indices.long()
		sizes = size[y_indices, x_indices]
		offsets = offset[y_indices, x_indices]

		sizes = torch.flip(sizes, [1])
		offsets = torch.flip(offsets, [1])

		pos = combined_indices + offsets

		zero_tensors = torch.zeros((top_k, 2))
		cur_device = sizes.device
		zero_tensors = zero_tensors.to(cur_device)

		#hight_width = torch.maximum(sizes, zero_tensors)
		hight_width = sizes / 2

		min_pos = pos - hight_width
		max_pos = pos + hight_width

		boxes = torch.cat([min_pos, max_pos], dim=1)
		boxes = torch.clip(boxes, 0., self.tensor_dim-1.)
		boxes = boxes * 4.

		return boxes, detection_classes, detection_scores
