import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Vgg19(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super(Vgg19, self).__init__()
		vgg_pretrained_cnn = models.vgg19(pretrained=True).features
		#self.vgg_pretrained_classifer = models.vgg19(pretrained=False).classifier

		self.conv1_1 = nn.Sequential(vgg_pretrained_cnn[0],vgg_pretrained_cnn[1])
		self.conv1_2 = nn.Sequential(vgg_pretrained_cnn[2],vgg_pretrained_cnn[3])

		self.conv2_1 = nn.Sequential(vgg_pretrained_cnn[5],vgg_pretrained_cnn[6])
		self.conv2_2 = nn.Sequential(vgg_pretrained_cnn[7],vgg_pretrained_cnn[8])

		self.conv3_1 = nn.Sequential(vgg_pretrained_cnn[10],vgg_pretrained_cnn[11])
		self.conv3_2 = nn.Sequential(vgg_pretrained_cnn[12],vgg_pretrained_cnn[13])
		self.conv3_3 = nn.Sequential(vgg_pretrained_cnn[14],vgg_pretrained_cnn[15])
		self.conv3_4 = nn.Sequential(vgg_pretrained_cnn[16],vgg_pretrained_cnn[17])

		self.conv4_1 = nn.Sequential(vgg_pretrained_cnn[19],vgg_pretrained_cnn[20])
		self.conv4_2 = nn.Sequential(vgg_pretrained_cnn[21],vgg_pretrained_cnn[22])
		self.conv4_3 = nn.Sequential(vgg_pretrained_cnn[23],vgg_pretrained_cnn[24])
		self.conv4_4 = nn.Sequential(vgg_pretrained_cnn[25],vgg_pretrained_cnn[26])

		self.conv5_1 = nn.Sequential(vgg_pretrained_cnn[27],vgg_pretrained_cnn[28],nn.ReflectionPad2d(1),vgg_pretrained_cnn[29])
		#self.conv5_2 = nn.Sequential(vgg_pretrained_cnn[30],nn.ReflectionPad2d(1),vgg_pretrained_cnn[31])
		#self.conv5_3 = nn.Sequential(vgg_pretrained_cnn[32],nn.ReflectionPad2d(1),vgg_pretrained_cnn[33])
		#self.conv5_4 = nn.Sequential(vgg_pretrained_cnn[34],nn.ReflectionPad2d(1),vgg_pretrained_cnn[35])


		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, x, out_keys):

		out = {}
		out['r11'] = self.conv1_1(x)
		out['r12'] = self.conv1_2(out['r11'])
		out['max12'], out['idx12'] = F.max_pool2d(out['r12'],kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)

		out['r21'] = self.conv2_1(out['max12'])
		out['r22'] = self.conv2_2(out['r21'])
		out['max22'], out['idx22'] = F.max_pool2d(out['r22'],kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)

		out['r31'] = self.conv3_1(out['max22'])
		out['r32'] = self.conv3_2(out['r31'])
		out['r33'] = self.conv3_3(out['r32'])
		out['r34'] = self.conv3_4(out['r33'])
		out['max34'], out['idx34'] = F.max_pool2d(out['r34'],kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)

		out['r41'] = self.conv4_1(out['max34'])
		#out['r42'] = self.conv4_2(out['r41'])
		#out['r43'] = self.conv4_3(out['r42'])
		#out['r44'] = self.conv4_4(out['r43'])

		#out['r51'] = self.conv5_1(out['r44'])
		#out['r52'] = self.conv5_2(out['r51'])
		#out['r53'] = self.conv5_3(out['r52'])
		#out['r54'] = self.conv5_4(out['r53'])

		return [out[key] for key in out_keys]

