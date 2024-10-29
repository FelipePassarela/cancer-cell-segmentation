import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.image_dataset import ImageDataset
from dataset.transforms import get_train_transforms, get_val_transforms
from models.deeplab_v3p import DeepLabV3Plus
from models.unet import UNet
from utils.metrics import BCEDiceLoss, update_history
from utils.utils import set_seed


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

N_EPOCHS = config["N_EPOCHS"]
LEARNING_RATE = config["LEARNING_RATE"]
BATCH_SIZE = config["BATCH_SIZE"]
NUM_WORKERS = config["NUM_WORKERS"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = config["TEST_DIR"]
VAL_DIR = config["VAL_DIR"]
TRAIN_DIR = config["TRAIN_DIR"]
MODEL_SAVE_DIR = config["MODEL_SAVE_DIR"]
WANDB_PROJECT = config["WANDB_PROJECT"]

wandb.init(project=WANDB_PROJECT, config=config)


def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: callable,
        device: str,
        scaler: torch.amp.GradScaler,
        epoch: int = None,
) -> dict[str, np.ndarray]:
    n_batches = len(dataloader)
    running_metrics = {
        "loss": 0.0,
        "dice": 0.0,
        "hausdorff": 0.0
    }
    history = {
        "loss": np.zeros(n_batches),
        "dice": np.zeros(n_batches),
        "hausdorff": np.zeros(n_batches)
    }
    progress = tqdm(enumerate(dataloader), total=n_batches, desc="Training", unit="batch")

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
        scheduler.step() if scheduler else None

        history, running_metrics = update_history(history, running_metrics, masks, preds, loss, i)
        wandb.log({
            "Train/Loss": history["loss"][i],
            "Train/Dice": history["dice"][i],
            "Train/Hausdorff": history["hausdorff"][i],
            "Train/LearningRate": scheduler.get_last_lr()[0] if scheduler else None,
        })

        progress.set_postfix({
            "loss": f"{history["loss"][i]:.4f}",
            "dice": f"{history["dice"][i]:.4f}",
        })

    return history


def eval_step(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: callable,
        device: str,
        epoch: int = None,
        test_set: bool = False,
) -> dict[str, np.ndarray]:
    n_batches = len(dataloader)
    running_metrics = {
        "loss": 0.0,
        "dice": 0.0,
        "hausdorff": 0.0
    }
    history = {
        "loss": np.zeros(n_batches),
        "dice": np.zeros(n_batches),
        "hausdorff": np.zeros(n_batches)
    }

    desc = "Testing" if test_set else "Validation"
    progress = tqdm(enumerate(dataloader), total=n_batches, desc=desc, unit="batch")

    model.eval()
    with torch.no_grad():
        for i, (imgs, masks) in progress:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            history, running_metrics = update_history(history, running_metrics, masks, preds, loss, i)

            if not test_set:
                wandb.log({
                    "Val/Loss": history["loss"][i],
                    "Val/Dice": history["dice"][i],
                    "Val/Hausdorff": history["hausdorff"][i],
                })

                if i == 0:
                    wandb.log({
                        "Val/Images": [wandb.Image(img) for img in imgs],
                        "Val/Masks": [wandb.Image(mask) for mask in masks],
                        "Val/Predictions": [wandb.Image(pred) for pred in preds],
                    })

            progress.set_postfix({
                "loss": f"{history["loss"][i]:.4f}",
                "dice": f"{history["dice"][i]:.4f}",
            })

    return history


def train(model: nn.Module):
    set_seed()
    model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    model_name = type(model).__name__
    model_name_ext = model_name + time.strftime("_%d-%m-%Y_%H-%M-%S")

    wandb.run.name = model_name_ext

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
    best_scores = {}
    best_model_path = ""

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{N_EPOCHS}] - {model_name}")
        print("-" * 30)

        train_hist = train_step(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            scaler,
            epoch,
        )
        val_hist = eval_step(
            model,
            val_loader,
            criterion,
            DEVICE,
            epoch,
        )

        train_loss = train_hist["loss"][-1]
        train_dice = train_hist["dice"][-1]
        train_hausdorff = train_hist["hausdorff"][-1]
        val_loss = val_hist["loss"][-1]
        val_dice = val_hist["dice"][-1]
        val_hausdorff = val_hist["hausdorff"][-1]

        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_scores = {
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_hausdorff": train_hausdorff,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_hausdorff": val_hausdorff,
            }
            best_model_path = (Path(MODEL_SAVE_DIR) / f"{model_name_ext}.pth").as_posix()

            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Model improved! - Saved at: {best_model_path}")

        print(f"lr: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Train Loss: {train_loss:.4f} - Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f}")

    print("-" * 30 + "\nTraining complete\n" + "-" * 30)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_hist = eval_step(model, test_loader, criterion, DEVICE, test_set=True)
    test_loss = test_hist["loss"][-1]
    test_dice = test_hist["dice"][-1]
    test_hausdorff = test_hist["hausdorff"][-1]
    print(f"Test Loss: {test_loss:.4f} - Test Dice: {test_dice:.4f}")

    wandb.log({
        'hparam/hp_model': model_name,
        'hparam/hp_optimizer': type(optimizer).__name__,
        'hparam/hp_scheduler': type(scheduler).__name__,
        'hparam/hp_learning_rate': LEARNING_RATE,
        'hparam/hp_batch_size': BATCH_SIZE,
        'hparam/hp_n_epochs': N_EPOCHS,

        'hparam/m_train_loss': best_scores["train_loss"],
        'hparam/m_train_dice': best_scores["train_dice"],
        'hparam/m_train_hausdorff': best_scores["train_hausdorff"],
        'hparam/m_val_loss': best_scores["val_loss"],
        'hparam/m_val_dice': best_scores["val_dice"],
        'hparam/m_val_hausdorff': best_scores["val_hausdorff"],
        'hparam/m_test_loss': test_loss,
        'hparam/m_test_dice': test_dice,
        'hparam/m_test_hausdorff': test_hausdorff,
    })

    wandb.finish()
    print()


if __name__ == "__main__":
    train(UNet(3, 1))
    train(DeepLabV3Plus(3, 1))
