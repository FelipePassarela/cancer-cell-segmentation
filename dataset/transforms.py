from typing import Tuple

import torch
import torchvision.transforms.v2 as transforms
import yaml


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

IMAGE_SIZE = tuple(config["IMAGE_SIZE"])


def get_train_transforms(size: Tuple[int, int] = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(size, antialias=True),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAutocontrast(),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToDtype(torch.float32, scale=True),
    ])


def get_val_transforms(size: Tuple[int, int] = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(size, antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
    ])


def get_inference_transforms():
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
    ])
