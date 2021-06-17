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


import torch.nn as nn
from model import UNetFeature, UNetFCN


class OgModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = UNetFeature()
        self.predict = UNetFCN(out_channels=1)

    def forward(self, left_image, right_image):
        featureL = self.feature(left_image)
        depth = self.predict(*featureL)

        return depth
