# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import json
from util import load_depth, load_image
import torch

from torch.utils.data import Dataset, DataLoader


def create_dataloader(dataset_root, json_path, batch_size=2, img_size=256, workers=8, pin_memory=True, shuffle=True):
    dataset = BDataset(dataset_root, json_path, img_size=img_size)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=pin_memory, shuffle=shuffle)
    return dataset, dataloader


class BDataset(Dataset):
    def __init__(self, dataset_root, json_path, img_size=256):
        super(BDataset, self).__init__()
        self.dataset_root = dataset_root
        self.json_path = os.path.join(dataset_root, json_path)
        self.img_size = img_size

        with open(self.json_path, "r") as f:
            self.json_data = json.load(f)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data = self.__load__(index)
        data = self.__transform__(data)
        return data

    def __load__(self, index):
        left_img_path = os.path.join(self.dataset_root, self.json_data[index]["imageL"])
        right_img_path = os.path.join(self.dataset_root, self.json_data[index]["imageR"])
        left_depth_path = os.path.join(self.dataset_root, self.json_data[index]["depthL"])
        right_depth_path = os.path.join(self.dataset_root, self.json_data[index]["depthR"])

        left_img = load_image(left_img_path)
        right_img = load_image(right_img_path)
        left_depth = load_depth(left_depth_path)
        right_depth = load_depth(right_depth_path)

        return left_img, right_img, left_depth, right_depth

    def __transform__(self, data):
        left_img, right_img, left_depth, right_depth = data
        
        left_img = left_img.transpose(2, 0, 1)
        right_img = right_img.transpose(2, 0, 1)
        left_depth = left_depth.transpose(2, 0, 1)
        right_depth = right_depth.transpose(2, 0, 1)
        
        return torch.from_numpy(left_img), torch.from_numpy(right_img), torch.from_numpy(left_depth), torch.from_numpy(right_depth)


class LoadImages():
    def __init__(self, json_data, img_size=256):
        self.json_data = json_data
        self.img_size = img_size
        self.count = 0
        
    def __len__(self):
        return len(self.json_data)
        
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        index = self.count
        
        if self.count == self.__len__():
            raise StopIteration
        self.count += 1
        
        data = self.__load__(index)
        data =  self.__transform__(data)
        return data
        
    def __load__(self, index):
        left_img_path = self.json_data[index]["imageL"]
        right_img_path = self.json_data[index]["imageR"]
        output_img_path = self.json_data[index]["output"]

        left_img = load_image(left_img_path)
        right_img = load_image(right_img_path)

        return left_img, right_img, output_img_path
    
    def __transform__(self, data):
        left_img, right_img, output_path = data
        
        left_img = left_img.transpose(2, 0, 1)
        right_img = right_img.transpose(2, 0, 1)
        
        return torch.from_numpy(left_img), torch.from_numpy(right_img), output_path


if __name__ == "__main__":
    from config import JSON
    _, dataloader = create_dataloader("../bdataset_stereo", "train.json")
    left_imgs, right_imgs, left_depths, right_depths = next(iter(dataloader))
    assert left_imgs.shape == right_imgs.shape, "dataset error"
    assert left_depths.shape == right_depths.shape, "dataset error"
    assert left_imgs.shape == (2, 3, 256, 256), "dataset error"
    assert left_depths.shape == (2, 1, 256, 256), "dataset error"
    
    dataset = LoadImages(JSON)
    left_img, right_img, path = next(iter(dataset))
    assert left_img.shape == (1, 3, 256, 256), "dataset error"
    assert right_img.shape == (1, 3, 256, 256), "dataset error"

    print("dataset ok")
