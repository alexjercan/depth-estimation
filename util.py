# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import cv2
import numpy as np

def load_image(path):
    img = img2bgr(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    return img


def load_depth(path):
    img = exr2depth(path)  # 1 channel depth
    assert img is not None, 'Image Not Found ' + path
    return img


def img2bgr(path):
    if not os.path.isfile(path):
        return None
    
    img = cv2.imread(path)
    
    img = img / 255
    
    img = np.array(img).astype(np.float32)
    
    return img


def exr2depth(path, maxvalue=80):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    img[img > maxvalue] = maxvalue
    img = img / maxvalue

    img = np.array(img).astype(np.float32).reshape((img.shape[0], img.shape[1], -1))

    return img