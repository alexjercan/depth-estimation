# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import albumentations as A
from albumentations.pytorch import ToTensorV2


class MyHorizontalFlip(A.HorizontalFlip):
    @property
    def targets(self):
        return dict(super().targets, **{'depth': self.apply, 'normal': self.apply_normal})

    def apply_normal(self, img, **params):
        # when flipping horizontally the normal map should be inversed on the x axis
        img[:, :, 0] = -1 * img[:, :, 0]
        return super().apply(img, **params)


class MyVerticalFlip(A.VerticalFlip):
    @property
    def targets(self):
        return dict(super().targets, **{'depth': self.apply, 'normal': self.apply_normal})

    def apply_normal(self, img, **params):
        img[:, :, 1] = -1 * img[:, :, 1]  # y axis flip for normal maps
        return super().apply(img, **params)


class MyRandomResizedCrop(A.RandomResizedCrop):
    @property
    def targets(self):
        return dict(super().targets, **{'depth': self.apply, 'normal': self.apply})


class MyOpticalDistortion(A.OpticalDistortion):
    @property
    def targets(self):
        return dict(super().targets, **{'depth': self.apply, 'normal': self.apply})


class MyGridDistortion(A.GridDistortion):
    @property
    def targets(self):
        return dict(super().targets, **{'depth': self.apply, 'normal': self.apply})


class MyIAAPiecewiseAffine(A.IAAPiecewiseAffine):
    @property
    def targets(self):
        return dict(super().targets, **{'depth': self.apply, 'normal': self.apply})


class MyToTensorV2(ToTensorV2):
    @property
    def targets(self):
        return dict(super().targets, **{'depth': self.apply, 'normal': self.apply})
