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

from collections import defaultdict
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
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CNNBlockT(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlockT, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpProjBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpProjBlock, self).__init__()
        aux_channels = out_channels // 4
        
        self.block1 = CNNBlockT(in_channels, aux_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        
        self.block2 = CNNBlock(aux_channels, aux_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(aux_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(aux_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.block1(x)
        
        y = self.block2(x)
        y = self.bn3(self.conv3(y))

        y = y + self.bn4(self.conv4(x))
        y = self.relu(y)
        
        return y


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
        self.relu = nn.LeakyReLU(0.2, inplace=True)
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
        self.relu = nn.LeakyReLU(0.2, inplace=True)
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
        self.relu = nn.LeakyReLU(0.2, inplace=True)
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
        x = self.layer3(x)
        x = self.layer4(x)

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


class DecoderFCN(nn.Module):
    def __init__(self, out_channels):
        super(DecoderFCN, self).__init__()
        self.layer1 = CNNBlock(2048, 1024, kernel_size=1, bias=False)
        self.layer2 = UpProjBlock(1024, 512)
        self.layer3 = UpProjBlock(512, 256)
        self.layer4 = UpProjBlock(256, 128)
        self.layer5 = UpProjBlock(128, 64)
        
        self.conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)

        y = self.conv(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)

        return y


class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.feature = resnet50(**kwargs)
        
        self.c_feature = CNNBlock(4096, 2048, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.depthFCN = DecoderFCN(1)
        self.normalsFCN = DecoderFCN(3)

    def forward(self, left_image, right_image):
        featureL = self.feature(left_image)
        featureR = self.feature(right_image)

        feature = torch.cat((featureL, featureR), dim=1)
        feature = self.c_feature(feature)

        depth = self.depthFCN(feature)
        norm = self.normalsFCN(feature)

        return depth, norm


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.depth_loss = nn.L1Loss(reduction='mean')
        self.normal_loss = nn.L1Loss(reduction='mean')

        self.item_losses = defaultdict(list)
        self.losses = []

    def forward(self, predictions, targets):
        (depth_p, normal_p) = predictions
        (depth_gt, normal_gt) = targets
                
        item_losses = {'depth': self.depth_loss(depth_p, depth_gt) * 1.0, 
                       'normal': self.normal_loss(normal_p, normal_gt) * 1.0}
        loss = sum(item_losses.values())
        
        for s, il in item_losses.items():
            self.item_losses[s].append(il.item())
        self.losses.append(loss.item())
        
        return loss
    
    def show(self):
        mean_item_losses = {s: sum(il) / len(il) for s, il in self.item_losses.items()}
        mean_item_losses = " ".join(f'{s}:{il:.4f}' for s, il in mean_item_losses.items())
        
        mean_loss = sum(self.losses) / len(self.losses)
        return f'(total:{mean_loss:.4f} {mean_item_losses})'


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, predictions, targets):        
        return torch.sqrt(self.mse(predictions, targets))


if __name__ == "__main__":
    left = torch.rand((4, 3, 256, 256))
    right = torch.rand((4, 3, 256, 256))    
    model = Model()
    pred = model(left, right)
    assert isinstance(pred, tuple), "Model"
    assert pred[0].shape == (4, 1, 256, 256), "Model"
    assert pred[1].shape == (4, 3, 256, 256), "Model"

    print("model ok")
