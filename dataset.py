# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import json
from copy import copy

from torch.utils.data import Dataset, DataLoader
from util import load_depth, load_image


def create_dataloader(dataset_root, json_path, batch_size=2, transform=None, workers=8, pin_memory=True, shuffle=True, max_depth=80):
    dataset = BDataset(dataset_root, json_path, transform=transform, max_depth=max_depth)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=pin_memory, shuffle=shuffle)
    return dataset, dataloader


class BDataset(Dataset):
    def __init__(self, dataset_root, json_path, transform=None, max_depth=80):
        super(BDataset, self).__init__()
        self.dataset_root = dataset_root
        self.json_path = os.path.join(dataset_root, json_path)
        self.transform = transform
        self.max_depth = max_depth

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
        left_depth = load_depth(left_depth_path, max_depth=self.max_depth)
        right_depth = load_depth(right_depth_path, max_depth=self.max_depth)

        return left_img, right_img, left_depth, right_depth

    def __transform__(self, data):
        left_img, right_img, left_depth, right_depth = data

        if self.transform is not None:
            augmentations = self.transform(image=left_img, right_img=right_img,
                                           left_depth=left_depth, right_depth=right_depth)
            left_img = augmentations["image"]
            right_img = augmentations["right_img"]
            left_depth = augmentations["left_depth"]
            right_depth = augmentations["right_depth"]

        return left_img, right_img, left_depth, right_depth

class LoadImages():
    def __init__(self, json_data, transform=None):
        self.json_data = json_data
        self.transform = transform
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
        img = copy(left_img)

        if self.transform is not None:
            augmentations = self.transform(image=left_img, right_img=right_img)

            left_img = augmentations["image"]
            right_img = augmentations["right_img"]

        return img, left_img, right_img, output_path


if __name__ == "__main__":
    from config import JSON, IMAGE_SIZE
    import albumentations as A
    import my_albumentations as M
    import matplotlib.pyplot as plt

    def visualize(image):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(image)
        plt.show()

    my_transform = A.Compose(
        [
            M.MyRandomResizedCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
            M.MyHorizontalFlip(p=0.5),
            M.MyVerticalFlip(p=0.1),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                M.MyOpticalDistortion(p=0.3),
                M.MyGridDistortion(p=0.1),
                M.MyIAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
        additional_targets={
            'right_img': 'image',
            'left_depth': 'depth',
            'right_depth': 'depth',
        }
    )

    img_transform = A.Compose(
        [
            A.Normalize(),
            M.MyToTensorV2(),
        ],
        additional_targets={
            'right_img': 'image',
            'left_depth': 'depth',
            'right_depth': 'depth',
        }
    )

    _, dataloader = create_dataloader("../bdataset_stereo", "train.json", transform=my_transform, max_depth=80)
    left_imgs, right_imgs, left_depths, right_depths = next(iter(dataloader))
    assert left_imgs.shape == right_imgs.shape, "dataset error"
    assert left_depths.shape == right_depths.shape, "dataset error"
    assert left_imgs.shape == (2, 3, 256, 256), f"dataset error {left_imgs.shape}"
    assert left_depths.shape == (2, 1, 256, 256), f"dataset error {left_depths.shape}"

    dataset = LoadImages(JSON, transform=img_transform)
    img, left_img, right_img, path = next(iter(dataset))
    assert img.shape == (256, 256, 3), f"dataset error {img.shape}"
    assert left_img.shape == (3, 256, 256), f"dataset error {left_img.shape}"
    assert right_img.shape == (3, 256, 256), f"dataset error {right_img.shape}"

    print("dataset ok")
