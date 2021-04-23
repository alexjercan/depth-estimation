# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import yaml
import torch
from tqdm import tqdm

from config import parse_test_config, DEVICE
from model import Model, LossFunction
from general import load_checkpoint
from dataset import create_dataloader


def test(model=None, config=None):
    torch.backends.cudnn.benchmark = True

    config = parse_test_config() if not config else config

    _, dataloader = create_dataloader(config.DATASET_ROOT, config.JSON_PATH, 
                                      batch_size=config.BATCH_SIZE, img_size=config.IMAGE_SIZE,
                                      workers=config.WORKERS, pin_memory=config.PIN_MEMORY, shuffle=config.SHUFFLE)

    if not model:
        model = Model()
        model = model.to(DEVICE)
        _, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    loss_fn = LossFunction()

    loop = tqdm(dataloader, leave=True)
    losses = []

    model.eval()

    for _, (left_img, right_img, left_depth, right_depth) in enumerate(loop):
        with torch.no_grad():
            left_img = left_img.to(DEVICE, non_blocking=True)
            right_img = right_img.to(DEVICE, non_blocking=True)
            left_depth = left_depth.to(DEVICE, non_blocking=True)
            right_depth = right_depth.to(DEVICE, non_blocking=True)

            predictions = model(left_img, right_img)
            loss = loss_fn(predictions, right_depth)
            
            losses.append(loss.item())

            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)


if __name__ == "__main__":
    with open("test.yaml", "r") as f:
        config = parse_test_config(yaml.load(f))
    test(config=config)
