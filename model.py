# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://arxiv.org/pdf/2003.14299.pdf
# - https://arxiv.org/pdf/1807.08865.pdf
# - https://github.com/meteorshowers/StereoNet-ActiveStereoNet

import torch
import torch.nn as nn
import torch.nn.functional as F

from cat_fms import cat_fms
from soft_argmin import FasterSoftArgmin


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CNNBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential(
            conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
        ) if stride != 1 else None

    def forward(self, x):
        identity = x

        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.act(x + identity)
        return x


class FeatureExtraction(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        in_channel = 3
        out_channel = 32

        downsample = []
        for _ in range(k):
            downsample.append(CNNBlock(in_channel, out_channel, kernel_size=5, stride=2, padding=2))
            in_channel = out_channel
        self.downsample = nn.Sequential(*downsample)

        self.residual_blocks = nn.Sequential(*[ResBlock(out_channel, out_channel) for _ in range(6)])
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = x
        output = self.downsample(output)
        output = self.residual_blocks(output)
        return self.conv(output)


class CostFilter(nn.Module):
    def __init__(self):
        super().__init__()
        in_channel = 64
        out_channel = 32

        filters = []
        for _ in range(4):
            filters.append(CNNBlock3D(in_channel, out_channel, kernel_size=3, stride=1, padding=1))
            in_channel = out_channel

        self.filter = nn.Sequential(*filters)
        self.conv3d = nn.Conv3d(out_channel, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.filter(x)
        return self.conv3d(x)


class DisparityRefinment(nn.Module):
    def __init__(self, k):
        super().__init__()
        
        self.scale = pow(2, k)
        self.conv =  nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x + self.conv(x)


class Model(nn.Module):
    def __init__(self, k=3, maxdisp=192):
        super().__init__()
        self.maxdisp = (maxdisp + 1) // pow(2, k)
        self.feature = FeatureExtraction(k)
        self.filter = CostFilter()
        self.predict = FasterSoftArgmin(self.maxdisp)
        self.refine = DisparityRefinment(k)

    def forward(self, left_image, right_image):
        ref_fm = self.feature(left_image)
        target_fm = self.feature(right_image)

        cost = cat_fms(ref_fm, target_fm, self.maxdisp)
        cost = self.filter(cost)
        cost = torch.squeeze(cost, 1)

        disp = self.predict(cost)
        pred = self.refine(disp)

        return pred


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, predictions, targets):
        return self.l1(predictions, targets)


if __name__ == "__main__":
    left = torch.rand((4, 3, 256, 256))
    model = Model()
    pred = model.forward(left, left)
    assert pred.shape == (4, 1, 256, 256), "model"
    print("model ok")
