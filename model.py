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

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CNNBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(0.2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            CNNBlock(in_channels, inter_channels, kernel_size=3, padding=1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class NormalsFCN(nn.Module):
    def __init__(self, **kwargs):
        super(NormalsFCN, self).__init__()
        self.feature = resnet50(**kwargs)
        self.predict = FCNHead(512, 3)
    
    def forward(self, x):
        fm = self.feature(x)
    
        norm = self.predict(fm)
        norm = F.interpolate(norm, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return norm


class DisparityFCN(nn.Module):
    def __init__(self, **kwargs):
        super(DisparityFCN, self).__init__()
        self.feature = resnet50(**kwargs)
        self.predict = FCNHead(1024, 1)

    def forward(self, left_image, right_image):
        ref_fm = self.feature(left_image)
        target_fm = self.feature(right_image)

        disp = torch.cat((ref_fm, target_fm), dim=1)

        disp = self.predict(disp)
        disp = F.interpolate(disp, size=left_image.shape[-2:], mode='bilinear', align_corners=False)

        return disp


class DepthRefinement(nn.Module):
    def __init__(self, k=3):
        super(DepthRefinement, self).__init__()
        self.block = CNNBlock(1, 8, kernel_size=3, stride=1, padding=1)
        self.filter = nn.Sequential(*[CNNBlock(8, 8, kernel_size=3, stride=1, padding=1) for _ in range(k)])
        self.predict = nn.Conv2d(8, 1, kernel_size=1)
        
    def forward(self, depth):
        out = self.block(depth)    
        out = self.filter(out)
        out = self.predict(out)
        return out
        

class NormalRefinement(nn.Module):
    def __init__(self, k=3):
        super(NormalRefinement, self).__init__()
        self.block = CNNBlock(3, 8, kernel_size=3, stride=1, padding=1)
        self.filter = nn.Sequential(*[CNNBlock(8, 8, kernel_size=3, stride=1, padding=1) for _ in range(k)])
        self.predict = nn.Conv2d(8, 3, kernel_size=1)
        
    def forward(self, norm):
        out = self.block(norm)
        out = self.filter(out)
        out = self.predict(out)
        return out
        

class Model(nn.Module):
    def __init__(self, k=3, **kwargs):
        super(Model, self).__init__()
        self.disparityFCN = DisparityFCN(**kwargs)
        self.normalsFCN = NormalsFCN(**kwargs)
        
        self.depthRefinement = DepthRefinement(k)
        self.normalRefinment = NormalRefinement(k)

    def forward(self, left_image, right_image):
        depth = self.disparityFCN(left_image, right_image)
        norm = self.normalsFCN(left_image)
        
        r_depth = self.depthRefinement(depth)
        r_norm = self.normalRefinment(norm)

        return depth, norm, r_depth, r_norm


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.depth_loss = nn.L1Loss(reduction='mean')
        self.normal_loss = nn.L1Loss(reduction='mean')
        self.r_depth_loss = nn.L1Loss(reduction='mean')
        self.r_normal_loss = nn.L1Loss(reduction='mean')

    def forward(self, predictions, targets):
        (depth_p, normal_p, r_depth_p, r_normal_p) = predictions
        (depth_gt, normal_gt) = targets
                
        depth_loss = self.depth_loss(depth_p, depth_gt) * 0.5
        r_depth_loss = self.r_depth_loss(r_depth_p, depth_gt) * 0.5
        normal_loss = self.normal_loss(normal_p, normal_gt) * 0.5
        r_normal_loss = self.r_depth_loss(r_normal_p, normal_gt) * 0.5
        
        return [depth_loss, r_depth_loss, normal_loss, r_normal_loss], depth_loss + r_depth_loss + normal_loss + r_normal_loss


if __name__ == "__main__":
    left = torch.rand((4, 3, 256, 256))
    right = torch.rand((4, 3, 256, 256))
    model = DisparityFCN()
    pred = model(left, right)
    assert pred.shape == (4, 1, 256, 256), "DisparityFCN"
    
    model = NormalsFCN()
    pred = model(left)
    assert pred.shape == (4, 3, 256, 256), "NormalsFCN"
    
    model = Model()
    pred = model(left, right)
    assert isinstance(pred, tuple), "Model"
    assert pred[0].shape == (4, 1, 256, 256), "Model"
    assert pred[1].shape == (4, 3, 256, 256), "Model"
    assert pred[2].shape == (4, 1, 256, 256), "Model"
    assert pred[3].shape == (4, 3, 256, 256), "Model"

    print("model ok")
