import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(Bottleneck, self).__init__()
		interChannels = 4 * growthRate
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, interChannels, 
			kernel_size = 1, bias = False)

		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, growthRate, 
			kernel_size = 3, padding = 1, bias = False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = self.conv2(F.relu(self.bn2(out)))
		out = torch.cat((x, out), dim = 1)
		return out

class SingleLayer(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(SingleLayer, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, growthRate, 
			kernel_size = 3, padding = 1, bias = False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = torch.cat((x, out), 1)
		return out

class Transition(nn.Module):
	def __init__(self, nChannels, nOutChannels):
		super(Transition, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, nOutChannels,
		 kernel_size = 1, bias = False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = F.avg_pool2d(out, 2)
		return out