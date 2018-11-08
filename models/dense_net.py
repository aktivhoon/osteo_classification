import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import DataLoader

from .dense_layers import *

import sys
import math

class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size = 3, padding = 1, bias = False)

        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)		
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 =  Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
<<<<<<< HEAD
=======
        print(nChannels)
>>>>>>> 58fe86efb8d6b036bd6bb6f3b183fb70f3957afe
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        output_size = out.shape
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), (output_size[2], output_size[3])))
        out = self.fc(out)
        return out
