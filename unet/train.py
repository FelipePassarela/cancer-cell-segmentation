import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.image_dataset import ImageDataset
from dataset.transforms import get_train_transforms, get_val_transforms
from unet.model import UNet
from utils.constants import (
    N_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    SEED
)
from utils.metrics import dice_score


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to {seed}")


def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: callable,
        device: str,
        scaler: torch.amp.GradScaler,
        scheduler: lr_scheduler.LRScheduler = None
) -> tuple[float, float]:
    running_loss = 0.0
    running_dice = 0.0
    n_batches = len(dataloader)
    progress = tqdm(enumerate(dataloader), total=n_batches, desc="Training", unit="batch")

    model.train()
    for i, (imgs, masks) in progress:
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.autocast(device):
            preds = model(imgs)
            loss = criterion(preds, masks)
            dice = dice_score(preds, masks)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() if scheduler else None

        running_loss += loss.item()
        running_dice += dice.item()
        progress.set_postfix({
            "loss": f"{running_loss / (i + 1):.4f}",
            "dice": f"{running_dice / (i + 1):.4f}",
        })

    return running_loss / n_batches, running_dice / n_batches


def eval_step(model, dataloader, criterion, device, desc="Validating") -> tuple[float, float]:
    running_loss = 0.0
    running_dice = 0.0
    n_batches = len(dataloader)
    progress = tqdm(enumerate(dataloader), total=n_batches, desc=desc, unit="batch")

    model.eval()
    with torch.no_grad():
        for i, (imgs, masks) in progress:
            imgs, masks = imgs.to(device), masks.to(device)
            pred = model(imgs)
            loss = criterion(pred, masks)
            dice = dice_score(pred, masks)
            running_loss += loss.item()
            running_dice += dice.item()
            progress.set_postfix({
                "loss": f"{running_loss / (i + 1):.4f}",
                "dice": f"{running_dice / (i + 1):.4f}",
            })

    return running_loss / n_batches, running_dice / n_batches


def train():
    set_seed()
    print(f"Using device: {DEVICE}")

    train_set = ImageDataset("../data/train", transforms=get_train_transforms())
    val_set = ImageDataset("../data/val", transforms=get_val_transforms())
    test_set = ImageDataset("../data/test", transforms=get_val_transforms())

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(3, 1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=N_EPOCHS
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler()

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{N_EPOCHS}]\n" + "-" * 30)

        train_loss, train_dice = train_step(model, train_loader, optimizer, criterion, DEVICE, scaler, scheduler)
        val_loss, val_dice = eval_step(model, val_loader, criterion, DEVICE)

        print(f"lr: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Train Loss: {train_loss:.4f} - Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f}")
    print("-" * 30 + "\nTraining complete\n" + "-" * 30)

    test_loss, test_dice = eval_step(model, test_loader, criterion, DEVICE, desc="Testing")
    print(f"Test Loss: {test_loss:.4f} - Test Dice: {test_dice:.4f}")

    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/UNet.pth")


if __name__ == "__main__":
    train()
