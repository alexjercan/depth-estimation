# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def load_image(path):
    img = img2rgb(path)  # RGB
    assert img is not None, 'Image Not Found ' + path
    return img


def load_depth(path):
    img = exr2depth(path, maxvalue=10)  # 1 channel depth
    assert img is not None, 'Image Not Found ' + path
    return img


def load_normal(path):
    img = exr2normal(path)  # 3 channel normal
    assert img is not None, 'Image Not Found ' + path
    return img


def img2rgb(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path)

    img = img / 255

    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def exr2depth(path, maxvalue=80):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    img[img > maxvalue] = maxvalue
    img = img / maxvalue

    img = np.array(img).astype(np.float32).reshape(
        (img.shape[0], img.shape[1], -1))

    return img


def exr2normal(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    img[img > 1] = 1
    img[img < 0] = 0
    
    img = np.array(img).astype(np.float32).reshape(img.shape[0], img.shape[1], -1)

    return img


def plot_predictions(predictions, paths):
    depth_ps, normal_ps = predictions
    depth_ps = depth_ps.cpu().numpy()
    normal_ps = normal_ps.cpu().numpy()

    for depth_p, normal_p, path in zip(depth_ps, normal_ps, paths):
        depth = depth_p.transpose(1, 2, 0)
        normal = normal_p.transpose(1, 2, 0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(path)
        ax1.axis('off')
        ax1.imshow(depth)
        ax2.axis('off')
        ax2.imshow(normal)
        plt.show()

def save_predictions(predictions, paths):
    depth_ps, normal_ps = predictions
    depth_ps = depth_ps.cpu().numpy()
    normal_ps = normal_ps.cpu().numpy()

    for depth_p, normal_p, path in zip(depth_ps, normal_ps, paths):
        depth = depth_p.transpose(1, 2, 0)
        normal = normal_p.transpose(1, 2, 0)

        depth_path = str(Path(path).with_suffix(".depth.exr"))
        normal_path = str(Path(path).with_suffix(".normal.exr"))

        cv2.imwrite(depth_path, depth)
        cv2.imwrite(normal_path, normal)
        
        plt.axis('off')
        plt.imshow(depth)
        plt.savefig(str(Path(path).with_suffix(".depth.png")))
        plt.imshow(normal)
        plt.savefig(str(Path(path).with_suffix(".normal.png")))
