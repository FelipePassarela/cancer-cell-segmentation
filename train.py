import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset.image_dataset import ImageDataset
from dataset.transforms import get_train_transforms, get_val_transforms
from models.deeplab_v3p import DeepLabV3Plus
from models.unet import UNet
from utils.metrics import BCEDiceLoss, dice_score, hausdorff_distance
from utils.utils import set_seed

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

N_EPOCHS = config["Training"]["N_EPOCHS"]
LEARNING_RATE = config["Training"]["LEARNING_RATE"]
BATCH_SIZE = config["Training"]["BATCH_SIZE"]
NUM_WORKERS = config["Training"]["NUM_WORKERS"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = config["Dirs"]["TEST_DIR"]
VAL_DIR = config["Dirs"]["VAL_DIR"]
TRAIN_DIR = config["Dirs"]["TRAIN_DIR"]
MODEL_SAVE_DIR = config["Dirs"]["MODEL_SAVE_DIR"]
WANDB_PROJECT = config["WANDB_PROJECT"]


def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: callable,
        device: str,
        scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    running_metrics = {
        "loss": 0.0,
        "dice": 0.0,
        "hausdorff": 0.0
    }
    n_batches = len(dataloader)
    progress = tqdm(enumerate(dataloader), total=n_batches, desc="Training", unit="batch", colour="green")

    model.train()
    for i, (imgs, masks) in progress:
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.autocast(device):
            preds = model(imgs)
            loss = criterion(preds, masks)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss["combined"]).backward()
        scaler.step(optimizer)
        scaler.update()

        running_metrics["loss"] += loss["combined"].item()
        running_metrics["dice"] += dice_score(preds, masks)
        running_metrics["hausdorff"] += hausdorff_distance(preds, masks)

        progress.set_postfix({
            "loss": f"{running_metrics["loss"] / (i + 1):.4f}",
            "dice": f"{running_metrics["dice"] / (i + 1):.4f}",
        })

    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
    return avg_metrics


def eval_step(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: callable,
        device: str,
        test_set: bool = False,
        return_images: bool = False,
) -> tuple[dict[str, float], dict[str, list[Any]]] | dict[str, float]:
    running_metrics = {
        "loss": 0.0,
        "dice": 0.0,
        "hausdorff": 0.0
    }
    images_dict = {
        "images": [],
        "masks": [],
        "preds": [],
        "logits": []
    }

    n_batches = len(dataloader)
    desc = "Testing" if test_set else "Validation"
    progress = tqdm(enumerate(dataloader), total=n_batches, desc=desc, unit="batch", colour="blue")

    model.eval()
    with torch.no_grad():
        for i, (imgs, masks) in progress:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            running_metrics["loss"] += loss["combined"].item()
            running_metrics["dice"] += dice_score(preds, masks)
            running_metrics["hausdorff"] += hausdorff_distance(preds, masks)

            if return_images and i == n_batches - 1:
                pred = [model.predict(img, device=device) for img in imgs[:5]]
                pred = {k: [v[k] for v in pred] for k in pred[0].keys()}
                images_dict["images"] = pred["img"]
                images_dict["masks"] = masks[:5]
                images_dict["preds"].extend(pred["pred"])
                images_dict["logits"].extend(pred["logits"])

            progress.set_postfix({
                "loss": f"{running_metrics["loss"] / (i + 1):.4f}",
                "dice": f"{running_metrics["dice"] / (i + 1):.4f}",
            })

    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}

    if return_images:
        return avg_metrics, images_dict
    return avg_metrics


def train(model: nn.Module):
    set_seed()
    model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    model_name = type(model).__name__
    model_name_ext = model_name + time.strftime("_%d-%m-%Y_%H-%M-%S")
    wandb.init(project=WANDB_PROJECT, config=config["Training"], name=model_name_ext)

    train_set = ImageDataset(TRAIN_DIR, transforms=get_train_transforms())
    val_set = ImageDataset(VAL_DIR, transforms=get_val_transforms())
    test_set = ImageDataset(TEST_DIR, transforms=get_val_transforms())

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, min_lr=1e-6, patience=5)
    criterion = BCEDiceLoss()
    scaler = torch.amp.GradScaler()

    best_val_dice = 0
    best_model_path = ""

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{N_EPOCHS}] - {model_name}")
        print("-" * 30)

        train_metrics = train_step(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_metrics, val_images = eval_step(model, val_loader, criterion, DEVICE, return_images=True)

        scheduler.step(val_metrics["dice"])
        last_lr = scheduler.get_last_lr()[0]

        # Save the current best model
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_model_path = str(Path(MODEL_SAVE_DIR) / f"{model_name_ext}.pth")

            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Model improved! - Saved at: {best_model_path}")

        wandb.log({
            "Train/Loss": train_metrics["loss"],
            "Train/Dice": train_metrics["dice"],
            "Train/Hausdorff": train_metrics["hausdorff"],
            "Val/Loss": val_metrics["loss"],
            "Val/Dice": val_metrics["dice"],
            "Val/Hausdorff": val_metrics["hausdorff"],
            "Images/Images": [wandb.Image(img) for img in val_images["images"]],
            "Images/Masks": [wandb.Image(mask) for mask in val_images["masks"]],
            "Images/Preds": [wandb.Image(pred) for pred in val_images["preds"]],
            "Images/Logits": [wandb.Image(logits) for logits in val_images["logits"]],
            "lr": last_lr,
        })

        print(f"lr: {last_lr :.6f}")
        print(f"Train Loss: {train_metrics["loss"]:.4f} - Train Dice: {train_metrics["dice"]:.4f}")
        print(f"Val Loss: {val_metrics["loss"]:.4f} - Val Dice: {val_metrics["dice"]:.4f}")

    print("-" * 30 + "\nTraining complete\n" + "-" * 30)

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    test_metrics = eval_step(model, test_loader, criterion, DEVICE, test_set=True)
    print(f"Test Loss: {test_metrics["loss"]:.4f} - Test Dice: {test_metrics["dice"]:.4f}")

    wandb.log({
        "Test/Loss": test_metrics["loss"],
        "Test/Dice": test_metrics["dice"],
        "Test/Hausdorff": test_metrics["hausdorff"],
    })

    wandb.finish()
    print()


if __name__ == "__main__":
    train(UNet(3, 1))
    train(DeepLabV3Plus(3, 1))
