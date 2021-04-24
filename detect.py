# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch
import argparse

from config import parse_detect_config, DEVICE, read_yaml_config
from model import Model
from util import save_predictions
from general import load_checkpoint
from dataset import LoadImages


def detect(model=None, config=None):
    torch.backends.cudnn.benchmark = True

    config = parse_detect_config() if not config else config

    dataset = LoadImages(config.JSON, img_size=config.IMAGE_SIZE)

    if not model:
        model = Model()
        model = model.to(DEVICE)
        _, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    for left_img, right_img, path in dataset:
        with torch.no_grad():
            left_img = left_img.to(DEVICE, non_blocking=True).unsqueeze(0)
            right_img = right_img.to(DEVICE, non_blocking=True).unsqueeze(0)

            predictions = model(left_img, right_img)
            
            save_predictions(predictions, [path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run inference on model')
    parser.add_argument('--detect', type=str, default="detect.yaml", help='detect config file')
    opt = parser.parse_args()

    config_detect = parse_detect_config(read_yaml_config(opt.detect))
    
    detect(config=config_detect)