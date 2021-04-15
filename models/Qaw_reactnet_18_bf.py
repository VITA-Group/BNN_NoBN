'''
React-birealnet-18(modified from resnet)

BN setting: remove all BatchNorm layers
Conv setting: replace conv2d with ScaledstdConv2d (add alpha beta each blocks)
Binary setting: only activation are binarized

'''


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from layers import *

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return ScaledStdConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return ScaledStdConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return HardBinaryScaledStdConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def binaryconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return HardBinaryScaledStdConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, alpha, beta, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.alpha = alpha
        self.beta = beta 

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = binaryconv3x3(inplanes, planes, stride=stride)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x
        x_in = x*self.beta 

        out = self.move0(x_in)
        out = self.binary_activation(out)
        out = self.binary_conv(out)

        if self.downsample is not None:
            residual = self.downsample(x_in)

        out = out*self.alpha + residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out

class BiRealNet(nn.Module):

    def __init__(self, block, layers, imagenet=True, alpha=0.2, num_classes=1000):
        super(BiRealNet, self).__init__()
        self.inplanes = 64

        if imagenet:
            self.conv1 = ScaledStdConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = ScaledStdConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

        expected_var = 1.0
        self.layer1, expected_var = self._make_layer(block, 64, layers[0], alpha, expected_var)
        self.layer2, expected_var = self._make_layer(block, 128, layers[1], alpha, expected_var, stride=2)
        self.layer3, expected_var = self._make_layer(block, 256, layers[2], alpha, expected_var, stride=2)
        self.layer4, expected_var = self._make_layer(block, 512, layers[3], alpha, expected_var, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, alpha, expected_var, stride=1):

        beta = 1. / expected_var ** 0.5
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                binaryconv1x1(self.inplanes, planes * block.expansion)
            )
            # Reset expected var at a transition block
            expected_var = 1.0

        layers = []
        layers.append(block(self.inplanes, planes, alpha, beta, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            beta = 1. / expected_var ** 0.5
            layers.append(block(self.inplanes, planes, alpha, beta))
            expected_var += alpha ** 2

        return nn.Sequential(*layers), expected_var

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def birealnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model



