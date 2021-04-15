'''
ReActNet(modified from MobileNetv1)

BN setting: remove all BatchNorm layers
Conv setting: original Conv2d
Binary setting: both activation and weight are binarized

'''


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np

from layers import *

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def binaryconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.move11 = LearnableBias(inplanes)
        self.binary_3x3= binaryconv3x3(inplanes, inplanes, stride=stride)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = binaryconv1x1(inplanes, planes)
        else:
            self.binary_pw_down1 = binaryconv1x1(inplanes, inplanes)
            self.binary_pw_down2 = binaryconv1x1(inplanes, inplanes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2,2)

    def forward(self, x):

        out1 = self.move11(x)

        out1 = self.binary_activation(out1)
        out1 = self.binary_3x3(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        out2 = self.binary_activation(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 += out1

        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

class reactnet(nn.Module):
    def __init__(self, num_classes=1000):
        super(reactnet, self).__init__()
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1))
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x






