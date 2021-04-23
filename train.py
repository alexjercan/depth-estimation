# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import re

import yaml
import torch
from tqdm import tqdm

from config import parse_train_config, DEVICE
from datetime import datetime as dt
from model import Model, LossFunction
# from test import test
from general import init_weights, save_checkpoint, load_checkpoint
from dataset import create_dataloader

def train_one_epoch(model, dataloader, loss_fn, solver, epoch_idx):
    loop = tqdm(dataloader, leave=True)
    losses = []

    for _, (left_img, right_img, left_depth, right_depth) in enumerate(loop):
        left_img = left_img.to(DEVICE, non_blocking=True)
        right_img = right_img.to(DEVICE, non_blocking=True)
        left_depth = left_depth.to(DEVICE, non_blocking=True)
        right_depth = right_depth.to(DEVICE, non_blocking=True)

        predictions = model(left_img, right_img)
        loss = loss_fn(predictions, left_depth)

        losses.append(loss.item())

        model.zero_grad()
        loss.backward()
        solver.step()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss, epoch=epoch_idx)


def train(config):
    torch.backends.cudnn.benchmark = True

    _, dataloader = create_dataloader(config.DATASET_ROOT, config.JSON_PATH, batch_size=config.BATCH_SIZE,
                                      img_size=config.IMAGE_SIZE, workers=config.WORKERS, pin_memory=config.PIN_MEMORY, shuffle=True)

    model = Model()
    model.apply(init_weights)
    solver = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, betas=config.BETAS)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(solver, milestones=config.MILESTONES, gamma=config.GAMMA)
    model = model.to(DEVICE)

    loss_fn = LossFunction()

    init_epoch = 0
    if config.CHECKPOINT_FILE and config.LOAD_MODEL:
        init_epoch, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    output_dir = os.path.join(config.OUT_PATH, re.sub("[^0-9a-zA-Z]+", "-", dt.now().isoformat()))

    for epoch_idx in range(init_epoch, config.NUM_EPOCHS):
        model.train()

        train_one_epoch(model, dataloader, loss_fn, solver, epoch_idx)
        lr_scheduler.step()

        # test(model)
        save_checkpoint(epoch_idx, model, output_dir)


if __name__ == "__main__":
    with open("train.yaml", "r") as f:
        config = parse_train_config(yaml.load(f))
    train(config)
