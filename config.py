# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import yaml
import torch
from attr_dict import AttrDict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = "../bdataset_stereo"
BATCH_SIZE = 4
IMAGE_SIZE = 256
WORKERS = 8
PIN_MEMORY = True

LEARNING_RATE = 0.001
MOMENTUM = 0.9
DAMPENING = 0.1
WEIGHT_DECAY = 0.0001

MILESTONES = [150]
GAMMA = .1

NUM_EPOCHS = 5
TEST = True
OUT_PATH = './runs'
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "normal.pth"

JSON = [{"imageL": "data/left.png", "imageR": "data/right.png", "output": "data/depth.exr"}]


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
    c.MOMENTUM = config.get("MOMENTUM", MOMENTUM)
    c.DAMPENING = config.get("DAMPENING", DAMPENING)
    c.WEIGHT_DECAY = config.get("WEIGHT_DECAY", WEIGHT_DECAY)
    
    c.MILESTONES = config.get("MILESTONES", MILESTONES)
    c.GAMMA = config.get("GAMMA", GAMMA)

    c.NUM_EPOCHS = config.get("NUM_EPOCHS", NUM_EPOCHS)
    c.TEST = config.get("TEST", TEST) 
    c.OUT_PATH = config.get("OUT_PATH", OUT_PATH)
    c.LOAD_MODEL = config.get("LOAD_MODEL", LOAD_MODEL)
    c.SAVE_MODEL = config.get("SAVE_MODEL", SAVE_MODEL)
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


def parse_detect_config(config=None):
    config = {} if not config else config
    c = AttrDict()

    c.JSON = config.get("JSON", JSON)
    c.IMAGE_SIZE = config.get("IMAGE_SIZE", IMAGE_SIZE)
    c.CHECKPOINT_FILE = config.get("CHECKPOINT_FILE", CHECKPOINT_FILE)
    
    return c


def read_yaml_config(path):
    if not os.path.isfile(path):
        return None
    else:
        with open(path, "r") as f:
            return yaml.safe_load(f)
