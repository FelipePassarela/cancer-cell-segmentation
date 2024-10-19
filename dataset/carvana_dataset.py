import os
from typing import Tuple

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors

from dataset.transforms import get_train_transforms


class CarvanaDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms: callable = None) -> None:
        super(CarvanaDataset, self).__init__()

        self.root = root
        self.transforms = transforms

        self.dirs = "train_images", "train_masks" if train else "val_images", "val_masks"
        self.images = os.listdir(os.path.join(self.root, self.dirs[0]))
        self.masks = os.listdir(os.path.join(self.root, self.dirs[1]))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.root, self.dirs[0], self.images[idx])
        mask_path = os.path.join(self.root, self.dirs[1], self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        with tv_tensors.set_return_type("TVTensor"):
            img = tv_tensors.Image(img)
            mask = tv_tensors.Mask(mask) / 255.0

        if self.transforms:
            img, mask = self.transforms(img, mask)

        return img, mask


def test_dataset():
    dataset = CarvanaDataset("data", True, transforms=get_train_transforms())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    for img, mask in dataloader:
        print(img, img.shape, img.dtype)
        print(mask, mask.shape, mask.unique(), mask.dtype)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0, 0].imshow(img[0].permute(1, 2, 0))
        axs[0, 0].set_title("Image 1")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(mask[0].squeeze(), cmap="gray")
        axs[0, 1].set_title("Mask 1")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(img[1].permute(1, 2, 0))
        axs[1, 0].set_title("Image 2")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(mask[1].squeeze(), cmap="gray")
        axs[1, 1].set_title("Mask 2")
        axs[1, 1].axis("off")

        plt.tight_layout()
        plt.show()
        break


if __name__ == "__main__":
    test_dataset()
