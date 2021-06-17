# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://arxiv.org/pdf/2003.14299.pdf
# - https://arxiv.org/pdf/1807.08865.pdf
# - https://xjqi.github.io/geonet.pdf
# - https://github.com/meteorshowers/StereoNet-ActiveStereoNet
# - https://github.com/xjqi/GeoNet
# - https://openaccess.thecvf.com/content_ECCV_2018/papers/Xinjing_Cheng_Depth_Estimation_via_ECCV_2018_paper.pdf
# - https://github.com/EPFL-VILAB/XTConsistency/blob/master/modules/unet.py
# - https://arxiv.org/pdf/1606.00373.pdf
# - https://arxiv.org/pdf/2010.06626v2.pdf
# - https://github.com/dontLoveBugs/FCRN_pytorch/blob/master/criteria.py#L37
#


import torch
import torch.nn as nn
from model import UNetFeature, UNetBlockT


class UNetFCN(nn.Module):
    def __init__(self, out_channels=3):
        super(UNetFCN, self).__init__()
        self.up_block1 = UNetBlockT(1024, 512, 512)
        self.up_block2 = UNetBlockT(512, 256, 256)
        self.up_block3 = UNetBlockT(256, 128, 128)
        self.up_block4 = UNetBlockT(128, 64, 64)
        self.up_block5 = UNetBlockT(64, 32, 32)
        self.up_block6 = UNetBlockT(32, 16, 16)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        x = self.up_block1(x6, x7)
        x = self.up_block2(x5, x)
        x = self.up_block3(x4, x)
        x = self.up_block4(x3, x)
        x = self.up_block5(x2, x)
        x = self.up_block6(x1, x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


class OgModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = UNetFeature()
        self.predict = UNetFCN(out_channels=1)

    def forward(self, left_image, right_image):
        featureL = self.feature(left_image)
        depth = self.predict(*featureL)

        return depth


if __name__ == "__main__":
    left = torch.rand((4, 3, 256, 256))
    right = torch.rand((4, 3, 256, 256))
    model = OgModel()
    pred = model(left, right)
    assert pred.shape == (4, 1, 256, 256), "Model"

    print("model ok")
