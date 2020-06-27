import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

class VNect(nn.Module):
	def __init__(self, in_channels):
		super(VNect, self).__init__()
		
		self.conv_block_1_a = nn.Sequential(nn.Conv2d(in_channels, 512, 1, 1),
											nn.BatchNorm2d(512),
											nn.ReLU(),										
											nn.Conv2d(512, 512, 3, 1),
											nn.BatchNorm2d(512),
											nn.ReLU(),
											nn.Conv2d(512, 1024, 1, 1),
											nn.BatchNorm2d(512))
		self.conv_block_1_b = nn.Sequential(nn.Conv2d(in_channels, 1024, 1, 1),
											nn.BatchNorm2d(1024))
		self.conv_block_2 = nn.Sequential(nn.Conv2d(1024, 256, 1, 1),
										  nn.BatchNorm2d(256),
										  nn.ReLU(),
										  SeparableConv2d(256, 128, 3, 1),
										  nn.BatchNorm2d(128),
										  nn.ReLU(),
										  nn.Conv2d(128, 256, 1, 1),
										  nn.BatchNorm2d(256),
										  nn.ReLU())
		self.deconv_block_1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 1),
											nn.BatchNorm2d(128),
											nn.ReLU())

	def forward(self):
		pass

class MoVNect(nn.Module):
	def __init__(self):
		super(MoVNect, self).__init__()

	def forward(self):
		pass

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x