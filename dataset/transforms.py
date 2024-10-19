from typing import Tuple

import torch
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors
from torchvision.transforms.functional import InterpolationMode

from utils.constants import IMAGE_SIZE


def clamp_image(x, image: tv_tensors.Image = None):
    """Clamp the image tensor between 0 and 1 to avoid floating point imprecision."""
    if image is None:
        # In this case, `x` is the image tensor
        return tv_tensors.Image(x.clamp(0, 1))

    if isinstance(image, tv_tensors.Image):
        return x, tv_tensors.wrap(image.clamp(0, 1))
    return x, image


def get_train_transforms(size: Tuple[int, int] = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(size, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(10, translate=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
        transforms.ToDtype(torch.float32, scale=True),
        clamp_image,
    ])


def get_val_transforms(size: Tuple[int, int] = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(size, antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        clamp_image,
    ])
