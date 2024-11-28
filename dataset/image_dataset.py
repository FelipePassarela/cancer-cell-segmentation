from pathlib import Path
from typing import Tuple

import torch
import torchvision
import wandb
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.io import decode_image

from dataset.transforms import get_val_transforms

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

WANDB_PROJECT = config["WANDB_PROJECT"]


class ImageDataset(Dataset):
    def __init__(self, root: str, transforms: callable = None) -> None:
        super(ImageDataset, self).__init__()

        self.root = Path(root)
        self.transforms = transforms
        self.items = [item for item in self.root.iterdir() if item.is_dir()]

        self.images = [item / "images" / f"{item.name}.png" for item in self.items]
        self.masks = [item / "masks" for item in self.items]
        assert len(self.images) == len(self.masks)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        masks_path = list(self.masks[idx].iterdir())

        img = decode_image(img_path.as_posix(), mode="RGB")
        mask = [decode_image(mask_path.as_posix(), mode="GRAY") for mask_path in masks_path]
        mask = self.join_masks(mask)

        with tv_tensors.set_return_type("TVTensor"):
            img = tv_tensors.Image(img)
            mask = tv_tensors.Mask(mask) / 255.0

        if self.transforms:
            img, mask = self.transforms(img, mask)

        return img, mask

    @staticmethod
    def join_masks(masks: list) -> torch.Tensor:
        """ Combines a list of masks into a single mask """
        masks = torch.stack(masks, dim=0)
        masks = masks.sum(dim=0)
        return masks


def test_dataset():
    wandb.init(project=WANDB_PROJECT)
    batch_size = 100

    dataset = ImageDataset(config["Dirs"]["TRAIN_DIR"], transforms=get_val_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    images, labels = next(iter(dataloader))
    features = images.view(batch_size, -1)
    wandb.log({"embedding": wandb.Image(features)})

    img_grid = torchvision.utils.make_grid(images)
    mask_grid = torchvision.utils.make_grid(labels)
    wandb.log({"images": [wandb.Image(img_grid)], "masks": [wandb.Image(mask_grid)]})


if __name__ == "__main__":
    test_dataset()
