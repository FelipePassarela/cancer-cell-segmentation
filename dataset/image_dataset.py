from pathlib import Path
from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.io import decode_image

from dataset.transforms import get_train_transforms


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
    dataset = ImageDataset("../data/train", transforms=get_train_transforms())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for i, (img, mask) in enumerate(dataloader):
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

        if i == 5:
            break


if __name__ == "__main__":
    test_dataset()
