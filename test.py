# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch
import argparse
import albumentations as A
import my_albumentations as M

from tqdm import tqdm
from metrics import MetricFunction, print_single_error
from config import parse_test_config, DEVICE, read_yaml_config
from model import Model, LossFunction
from general import load_checkpoint
from dataset import create_dataloader


def run_test(model, dataloader, loss_fn, metric_fn):
    loop = tqdm(dataloader, position=0, leave=True)
    
    for _, (left_img, right_img, left_depth, right_depth, left_normal, right_normal) in enumerate(loop):
        with torch.no_grad():
            left_img = left_img.to(DEVICE, non_blocking=True)
            right_img = right_img.to(DEVICE, non_blocking=True)
            left_depth = left_depth.to(DEVICE, non_blocking=True)
            right_depth = right_depth.to(DEVICE, non_blocking=True)
            left_normal = left_normal.to(DEVICE, non_blocking=True)
            right_normal = right_normal.to(DEVICE, non_blocking=True)

            predictions = model(left_img, right_img)
            loss_fn(predictions, (left_depth, left_normal))
            metric_fn.evaluate(predictions, (left_depth, left_normal))

    loop.close()


def test(model=None, config=None):
    epoch = 0
    torch.backends.cudnn.benchmark = True

    config = parse_test_config() if not config else config

    transform = A.Compose(
        [
            M.MyToTensorV2(),
        ],
        additional_targets={
            'right_img': 'image',
            'left_depth': 'depth',
            'right_depth': 'depth',
            'left_normal': 'normal',
            'right_normal': 'normal',
        }
    )

    _, dataloader = create_dataloader(config.DATASET_ROOT, config.JSON_PATH,
                                      batch_size=config.BATCH_SIZE, transform=transform,
                                      workers=config.WORKERS, pin_memory=config.PIN_MEMORY, shuffle=config.SHUFFLE)

    if not model:
        model = Model()
        model = model.to(DEVICE)
        epoch, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    loss_fn = LossFunction()
    metric_fn = MetricFunction(config.BATCH_SIZE)

    model.eval()
    run_test(model, dataloader, loss_fn, metric_fn)
    print_single_error(epoch, loss_fn.show(), metric_fn.show())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--test', type=str, default="test.yaml", help='test config file')
    opt = parser.parse_args()

    config_test = parse_test_config(read_yaml_config(opt.test))

    test(config=config_test)
