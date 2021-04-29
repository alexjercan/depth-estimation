# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import random
import numpy as np
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F


class MyHorizontalFlip(DualTransform):
    def __call__(self, *args, force_apply, **kwargs):
        if (random.random() < self.p) or self.always_apply or force_apply:
            for key, img in kwargs.items():
                if "normal" in key:
                    img = self.apply_normal(img)
                else:
                    img = self.apply_img(img)
        return kwargs

    def apply_normal(self, img, **params):
        # when flipping horizontally the normal map should be inversed on the x axis
        img[:, :, 0] = -1 * img[:, :, 0] + 1
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return F.hflip_cv2(img)

        return F.hflip(img)

    def apply_img(self, img, **params):
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return F.hflip_cv2(img)

        return F.hflip(img)


class MyVerticalFlip(DualTransform):
    def __call__(self, *args, force_apply, **kwargs):
        if (random.random() < self.p) or self.always_apply or force_apply:
            for key, img in kwargs.items():
                if "normal" in key:
                    img = self.apply_normal(img)
                else:
                    img = self.apply_img(img)
        return kwargs

    def apply_normal(self, img, **params):
        img[:, :, 1] = -1 * img[:, :, 1] + 1  # y axis flip for normal maps
        return F.vflip(img)

    def apply_img(self, img, **params):
        return F.vflip(img)
