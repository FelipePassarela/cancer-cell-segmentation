import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import CarvanaDataset
from model import UNET
from utils.metrics import dice_score
from utils.transforms import get_train_transforms
from utils.constants import (
    N_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE
)


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
            "loss": f"{running_loss/(i+1):.4f}",
            "dice": f"{running_dice/(i+1):.4f}",
        })

    return running_loss / n_batches, running_dice / n_batches


def val_step(model, dataloader, criterion, device) -> tuple[float, float]:
    running_loss = 0.0
    running_dice = 0.0
    n_batches = len(dataloader)
    progress = tqdm(enumerate(dataloader), total=n_batches, desc="Validating", unit="batch")

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
                "loss": f"{running_loss/(i+1):.4f}",
                "dice": f"{running_dice/(i+1):.4f}",
            })

    return running_loss / n_batches, running_dice / n_batches


def main():
    print(f"Using device: {DEVICE}")

    full_set = CarvanaDataset("data", True, transforms=get_train_transforms())
    train_size = int(0.8 * len(full_set))
    test_size = len(full_set) - train_size
    train_set, test_set = random_split(full_set, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = UNET(3, 1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler()

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{N_EPOCHS}]\n" + "-" * 30)

        train_loss, train_dice = train_step(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_dice = val_step(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        print(f"lr: {scheduler.get_last_lr()[0]}")
        print(f"Train Loss: {train_loss:.4f} - Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/UNET_caranvas.pth")


if __name__ == "__main__":
    main()
