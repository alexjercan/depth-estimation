# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import re

import torch
import torch.optim
import argparse
from tqdm import tqdm

from config import LEARNING_RATE, parse_test_config, parse_train_config, DEVICE, read_yaml_config
from datetime import datetime as dt
from model import Model, LossFunction
from test import test
from general import init_weights, save_checkpoint, load_checkpoint
from dataset import create_dataloader


def train_one_epoch(model, dataloader, loss_fn, solver, epoch_idx):
    loop = tqdm(dataloader, leave=True)
    losses = []
    item_losses = []

    for _, (left_img, right_img, left_depth, right_depth, left_normal, right_normal) in enumerate(loop):
        left_img = left_img.to(DEVICE, non_blocking=True)
        right_img = right_img.to(DEVICE, non_blocking=True)
        left_depth = left_depth.to(DEVICE, non_blocking=True)
        right_depth = right_depth.to(DEVICE, non_blocking=True)
        left_normal = left_normal.to(DEVICE, non_blocking=True)
        right_normal = right_normal.to(DEVICE, non_blocking=True)

        predictions = model(left_img, right_img)
        item_loss, loss = loss_fn(predictions, (left_depth, left_normal))

        losses.append(loss.item())
        item_losses.append([il.item() for il in item_loss])

        model.zero_grad()
        loss.backward()
        solver.step()

        mean_loss = sum(losses) / len(losses)
        mean_item_loss = [sum(il) / len(losses) for il in zip(*item_losses)]
        loop.set_postfix(loss=mean_loss, item_losses=mean_item_loss, epoch=epoch_idx)


def train(config=None, config_test=None):
    torch.backends.cudnn.benchmark = True
    
    config = parse_train_config() if not config else config

    _, dataloader = create_dataloader(config.DATASET_ROOT, config.JSON_PATH, 
                                      batch_size=config.BATCH_SIZE, img_size=config.IMAGE_SIZE, 
                                      workers=config.WORKERS, pin_memory=config.PIN_MEMORY, shuffle=config.SHUFFLE)

    model = Model()
    model.apply(init_weights)
    solver = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=config.LEARNING_RATE, betas=config.BETAS, 
                              eps=config.EPS, weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(solver, milestones=config.MILESTONES, gamma=config.GAMMA)
    model = model.to(DEVICE)

    loss_fn = LossFunction()

    epoch_idx = 0
    if config.CHECKPOINT_FILE and config.LOAD_MODEL:
        epoch_idx, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    output_dir = os.path.join(config.OUT_PATH, re.sub("[^0-9a-zA-Z]+", "-", dt.now().isoformat()))

    for epoch_idx in range(epoch_idx, config.NUM_EPOCHS):
        model.train()

        train_one_epoch(model, dataloader, loss_fn, solver, epoch_idx)
        lr_scheduler.step()

        if config.TEST:
            test(model, config_test)
        if config.SAVE_MODEL:
            save_checkpoint(epoch_idx, model, output_dir)

    if not config.TEST:
        test(model, config_test)
    if not config.SAVE_MODEL:
        save_checkpoint(epoch_idx, model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--train', type=str, default="train.yaml", help='train config file')
    parser.add_argument('--test', type=str, default="test.yaml", help='test config file')
    opt = parser.parse_args()
    
    config_train = parse_train_config(read_yaml_config(opt.train))
    config_test = parse_test_config(read_yaml_config(opt.test))

    train(config_train, config_test)
