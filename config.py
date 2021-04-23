# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch
from attr_dict import AttrDict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = "../bdataset_stereo"
BATCH_SIZE = 4
IMAGE_SIZE = 256
WORKERS = 8
PIN_MEMORY = True

LEARNING_RATE = 1e-3
BETAS = [.9, .999]
MILESTONES = [150]
GAMMA = .5

NUM_EPOCHS = 5
OUT_PATH = './runs'
LOAD_MODEL = False
CHECKPOINT_FILE = "normal.pth"

def parse_train_config(config=None):
    config = {} if not config else config
    c = AttrDict()
    
    c.DATASET_ROOT = config.get("DATASET_ROOT", DATASET_ROOT)
    c.JSON_PATH = config.get("JSON_PATH", "train.json")
    c.BATCH_SIZE = config.get("BATCH_SIZE", BATCH_SIZE)
    c.IMAGE_SIZE = config.get("IMAGE_SIZE", IMAGE_SIZE)
    c.WORKERS = config.get("WORKERS", WORKERS)
    c.PIN_MEMORY = config.get("PIN_MEMORY", PIN_MEMORY)
    c.SHUFFLE = config.get("SHUFFLE", True)
    
    c.LEARNING_RATE = config.get("LEARNING_RATE", LEARNING_RATE)
    c.BETAS = config.get("BETAS", BETAS)
    c.MILESTONES = config.get("MILESTONES", MILESTONES)
    c.GAMMA = config.get("GAMMA", GAMMA)
    
    c.NUM_EPOCHS = config.get("NUM_EPOCHS", NUM_EPOCHS)
    c.OUT_PATH = config.get("OUT_PATH", OUT_PATH)
    c.LOAD_MODEL = config.get("LOAD_MODEL", LOAD_MODEL)
    c.CHECKPOINT_FILE = config.get("CHECKPOINT_FILE", CHECKPOINT_FILE)
    
    return c

def parse_test_config(config=None):
    config = {} if not config else config
    c = AttrDict()
    
    c.DATASET_ROOT = config.get("DATASET_ROOT", DATASET_ROOT)
    c.JSON_PATH = config.get("JSON_PATH", "test.json")
    c.BATCH_SIZE = config.get("BATCH_SIZE", BATCH_SIZE)
    c.IMAGE_SIZE = config.get("IMAGE_SIZE", IMAGE_SIZE)
    c.WORKERS = config.get("WORKERS", WORKERS)
    c.PIN_MEMORY = config.get("PIN_MEMORY", PIN_MEMORY)
    c.SHUFFLE = config.get("SHUFFLE", False)
    
    c.OUT_PATH = config.get("OUT_PATH", OUT_PATH)
    c.LOAD_MODEL = config.get("LOAD_MODEL", True)
    c.CHECKPOINT_FILE = config.get("CHECKPOINT_FILE", CHECKPOINT_FILE)
    
    return c