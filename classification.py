import torch
import torch.nn as nn
import torch.optim as optim

import copy
import glob
import sys
import math
import yaml
import ujson
import cv2
import os
from tqdm import tqdm
from termcolor import colored


import matplotlib.pyplot as plt
import numpy as np

from turbojpeg import TJPF_RGB, TurboJPEG


sys.path.append('./src')
sys.path.append('./src/KHI')
sys.path.append('./src/KHI/network')
sys.path.append('./src/KHI/network/pytorch')
sys.path.append('./src/KHI/network/tensorflow')

from src.KHI.network.pytorch.nn_torch import *

class TrafficLightColor(nn.Module):
	def __init__(self, root, image_size):
		super().__init__()
		
		self.images_dir = root
		self.images = glob.glob(root + '/*.jpg')
		self.annoatation_dir = root + '/color_annotation.json'

		self.color_annotation = ujson.load(open(self.annoatation_dir, 'r'))['annotations']

		self.len = len(self.images)

		self.turbo_jpeg = TurboJPEG()

		self.image_size = image_size

	def __len__(self):
		return self.len

	def __getitem__(self, idx):

		#image = cv2.imread(self.images[idx])
		in_file = open(self.images[idx], 'rb')
		image = self.turbo_jpeg.decode(in_file.read(), pixel_format=TJPF_RGB)
		image = cv2.resize(image, self.image_size)
		image = image.transpose(2,0,1).astype(np.float32)
		image = (2.0 / 255.0) * image - 1.0

		label = self.color_annotation[idx]['color'] - 1

		#data = {
		#	'images': image,
		#	'annotations': label,
		#}
		return image, label



class ClassificationNet(nn.Module):
	def __init__(self, num_class=7) -> None:
		super().__init__()
		
		self.act = ReLU()

		self.conv_model = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
			torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
			torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
			torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, bias=True),

			torch.nn.Conv2d(in_channels=16, out_channels=7, kernel_size=1, stride=1, padding=0, bias=True),
		)

		self.sep_conv_model = torch.nn.Sequential(

			torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),

			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32),
			torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),

			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True, groups=64),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),

			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True, groups=64),
			torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),

			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32),
			torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True),

			torch.nn.Conv2d(in_channels=16, out_channels=7, kernel_size=1, stride=1, padding=0, bias=True),
		)

		# 48,16
		# 24, 8
		# 12, 4
		# 6, 2
		# 3, 1
		self.sep_conv_model_1 = torch.nn.Sequential(

			torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
			torch.nn.ReLU(),

			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32),
			torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
			torch.nn.ReLU(),

			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True, groups=64),
			torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
			torch.nn.ReLU(),

			#torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True, groups=64),
			#torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
			#torch.nn.ReLU(),

			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32),
			torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True),
			torch.nn.ReLU(),

			torch.nn.Conv2d(in_channels=16, out_channels=7, kernel_size=1, stride=1, padding=0, bias=True),
		)





		self.dense_conv_model = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),

			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32),
			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),

			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True, groups=32),
			torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),

			
		)

		self.fc1 = torch.nn.Linear(32*4*4, 7)
		#self.softmax = torch.log_softmax(dim=1)

	def forward(self, x):


		x = self.sep_conv_model_1(x)
		x = torch.flatten(x, 1)
		#x = self.fc1(x)
		#x = torch.log_softmax(x, dim=1)
		
		return x




if __name__=='__main__':

	batch_size = 16

	train_root = 'E:/carvi_dataset/LearningDataset/20211215/dataset1/tl_dataset1/tl_color_data_original_size/train'
	eval_root = 'E:/carvi_dataset/LearningDataset/20211215/dataset1/tl_dataset1/tl_color_data_original_size/eval'
	image_size = (16,16)

	trainset = TrafficLightColor(root=train_root, image_size=image_size)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

	evalset = TrafficLightColor(root=eval_root, image_size=image_size)
	evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True, num_workers=0)

	classes = ('none', 'red', 'green', 'yellow', 'blink', 'green-left', 'red-left', 'red-yellow')

	def imshow(img):
		#img = img / 2 + 0.5     # unnormalize
		img = img / 255.
		npimg = img.numpy()[0]
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()


	net = ClassificationNet(7)

	import torchsummary
	summary_model = copy.deepcopy(net).cpu()
	summary_model = summary_model.eval()
	torchsummary.summary(summary_model, (3, 16, 16), 1, device='cpu')

	from torchsummaryX import summary
	summary(summary_model, torch.rand((1, 3, 16, 16)))

	#criterion = nn.CrossEntropyLoss()
	criterion = nn.NLLLoss()

	#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=0.0004)



	for epoch in range(10):
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data

			optimizer.zero_grad()

			outputs = net(inputs)
			outputs = torch.log_softmax(outputs, dim=1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 10 == 9:
				print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 10))
				running_loss = 0.0

	print('Finished Training')


	fn = str(epoch).zfill(5)
	PATH = 'E:/vscode/Torch/MultiNet_OD_custom/src/KHI/utils/' + f'dense_epoch-{fn}.pth'
	torch.save(net.state_dict(), PATH)

	net = net.eval()

	with torch.no_grad():
		correct = 0
		total = 0
		for data in evalloader:
			inputs, labels = data
			prediction = net(inputs)
			correct_prediction = torch.argmax(prediction, 1) == labels
			total += len(labels)
			correct += correct_prediction.sum().item()
			
		print('Test Accuracy: ', 100.*correct/total, '%')




# for i, data in enumerate(evalloader, 0):
# 	with torch.no_grad():
# 		inputs, labels = data
# 		prediction = net(data)
# 		correct_prediction = torch.argmax(prediction, 1) == labels
# 		accuracy = correct_prediction.float().mean()

# 	print('Accuracy:', accuracy.item())
