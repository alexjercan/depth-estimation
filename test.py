# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

from metrics import avg_error, metric_function, print_single_error
import torch
import argparse
from tqdm import tqdm

from config import parse_test_config, DEVICE, read_yaml_config
from model import Model, LossFunction
from general import load_checkpoint
from dataset import create_dataloader


def test(model=None, config=None):
    epoch = 0
    torch.backends.cudnn.benchmark = True

    config = parse_test_config() if not config else config

    _, dataloader = create_dataloader(config.DATASET_ROOT, config.JSON_PATH, 
                                      batch_size=config.BATCH_SIZE, img_size=config.IMAGE_SIZE,
                                      workers=config.WORKERS, pin_memory=config.PIN_MEMORY, shuffle=config.SHUFFLE)

    if not model:
        model = Model()
        model = model.to(DEVICE)
        epoch, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    loss_fn = LossFunction()
    
    total_step_val = 0
    error_sum_val = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                     'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                     'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0}
    error_avg = {'MSE':0, 'RMSE':0, 'ABS_REL':0, 'LG10':0, 'MAE':0,\
                 'DELTA1.02':0, 'DELTA1.05':0, 'DELTA1.10':0, \
                 'DELTA1.25':0, 'DELTA1.25^2':0, 'DELTA1.25^3':0}

    loop = tqdm(dataloader, leave=True)

    model.eval()

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
            error_result = metric_function(predictions, (left_depth, left_normal))
            
            total_step_val += left_img.shape[0]
            error_avg = avg_error(error_sum_val, error_result, total_step_val, left_img.shape[0])

    loop.close()
    print_single_error(epoch, 0, loss_fn.show(), error_avg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--test', type=str, default="test.yaml", help='test config file')
    opt = parser.parse_args()

    config_test = parse_test_config(read_yaml_config(opt.test))
    
    test(config=config_test)
