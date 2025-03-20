import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Normalize

class AttributeNet(nn.Module):
    def __init__(self, layers=5, patch_size=8, channels=3, normalize=None):
        super(AttributeNet, self).__init__()
        self.layers = layers
        self.patch_size = patch_size
        self.channels = channels

        self.pooling = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        if self.layers == 5 and self.channels == 3:
            self.conv6 = nn.Conv2d(64, 3, 3, 1, 1)
        elif self.layers == 6:
            self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
            self.bn5 = nn.BatchNorm2d(128)
            self.relu5 = nn.ReLU(inplace=True)

            if self.channels == 3:
                self.conv6 = nn.Conv2d(128, 3, 3, 1, 1)
        self.normalize = normalize

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        if self.patch_size in [2, 4, 8, 16, 32]:
            y = self.pooling(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        if self.patch_size in [4, 8, 16, 32]:
            y = self.pooling(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        if self.patch_size in [8, 16, 32]:
            y = self.pooling(y)
        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu4(y)
        if self.patch_size in [16, 32]:
            y = self.pooling(y)
        if self.layers == 6:
            y = self.conv5(y)
            y = self.bn5(y)
            y = self.relu5(y)
            if self.patch_size == 32:
                y = self.pooling(y)

        if self.channels == 3:
            y = self.conv6(y)
        elif self.channels == 1:
            y = torch.mean(y, dim=1)
        return y

class InstancewiseVisualPrompt(nn.Module):
    def __init__(self, size, layers=5, patch_size=8, channels=3, normalize=None):
        '''

        Args:
            size: input image size
            layers: the number of layers of mask-training CNN
            patch_size: the size of patches with the same mask value
            channels: 3 means that the mask value for RGB channels are different, 1 means the same
            keep_watermark: whether to keep the reprogram (\delta) in the model
        '''
        super(InstancewiseVisualPrompt, self).__init__()
        if layers not in [5, 6]:
            raise ValueError("Input layer number is not supported")
        if patch_size not in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Input patch size is not supported")
        if channels not in [1, 3]:
            raise ValueError("Input channel number is not supported")
        if patch_size == 32 and layers != 6:
            raise ValueError("Input layer number and patch size are conflict with each other")

        # Set the attribute mask CNN
        self.patch_num = int(size / patch_size)
        self.imagesize = size
        self.patch_size = patch_size
        self.channels = channels
        self.priority = AttributeNet(layers, patch_size, channels, normalize=normalize)

        self.up_sampler = torch.nn.UpsamplingBilinear2d((size, size))

        # Set reprogram (\delta) according to the image size
        self.size = size
        self.program = torch.nn.Parameter(data=torch.zeros((3, size, size)))

    def forward(self, x):
        x = self.up_sampler(x)
        attention = self.priority(x).view(-1, self.channels, self.patch_num * self.patch_num, 1).expand(-1, 3, -1, self.patch_size * self.patch_size).view(-1, 3, self.patch_num, self.patch_num, self.patch_size, self.patch_size).transpose(3, 4)
        attention = attention.reshape(-1, 3, self.imagesize, self.imagesize)
        x = x + self.program * attention
        return x
    


class InputNormalize(nn.Module):
    def __init__(self, model, new_mean=(0.4914, 0.4822, 0.4465), new_std=(0.2471, 0.2435, 0.2616)):
        super(InputNormalize, self).__init__()
        new_mean = torch.tensor(new_mean)[..., None, None]
        new_std = torch.tensor(new_std)[..., None, None]
        self.register_buffer('new_mean', new_mean)
        self.register_buffer('new_std', new_std)
        self.model = model

    def forward(self, x):
        x = (x - self.new_mean) / self.new_std
        return self.model(x)
    

class ExpansiveVisualPrompt(nn.Module):
    def __init__(self, out_size, mask, init = 'zero'):
        super(ExpansiveVisualPrompt, self).__init__()
        assert mask.shape[0] == mask.shape[1]
        in_size = mask.shape[0]
        self.out_size = out_size
        if init == "zero":
            self.program = torch.nn.Parameter(data=torch.zeros(3, out_size, out_size)) 
        elif init == "randn":
            self.program = torch.nn.Parameter(data=torch.randn(3, out_size, out_size)) 
        else:
            raise ValueError("init method not supported")

        self.l_pad = int((out_size-in_size+1)/2)
        self.r_pad = int((out_size-in_size)/2)

        mask = np.repeat(np.expand_dims(mask, 0), repeats=3, axis=0)
        mask = torch.Tensor(mask)
        self.register_buffer("mask", F.pad(mask, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=1))

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0) + torch.sigmoid(self.program) * self.mask
        return x


class FullyVisualPrompt(nn.Module):
    def __init__(self, size):
        super(FullyVisualPrompt, self).__init__()

        self.size = size
        self.program = torch.nn.Parameter(data=torch.zeros(3, size, size)) 
        self.up_sampler = torch.nn.UpsamplingBilinear2d((size, size))

    def forward(self, x):
        x = self.up_sampler(x)
        x = x + self.program
        return x
    
    