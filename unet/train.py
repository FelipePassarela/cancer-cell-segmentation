import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    IMAGE_SIZE,
    set_seed
)
from utils.metrics import BCEDiceLoss, hausdorff_distance, dice_score


def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: callable,
        device: str,
        scaler: torch.amp.GradScaler,
        epoch: int = None,
        writer: SummaryWriter = None,
        scheduler: lr_scheduler.LRScheduler = None,
) -> dict[str, np.ndarray]:
    running_loss = 0.0
    running_dice = 0.0
    running_hausdorff = 0.0
    n_batches = len(dataloader)
    history = {"loss": np.zeros(n_batches), "dice": np.zeros(n_batches), "hausdorff": np.zeros(n_batches)}
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

        history["loss"][i] = loss["combined"].item()
        history["dice"][i] = dice_score(preds, masks)
        history["hausdorff"][i] = hausdorff_distance(preds, masks)

        if writer is not None:
            global_step = epoch * n_batches + i
            last_lr = scheduler.get_last_lr()[0]

            writer.add_scalar("Loss/train", history["loss"][i], global_step)
            writer.add_scalar("Dice/train", history["dice"][i], global_step)
            writer.add_scalar("Hausdorff/train", history["hausdorff"][i], global_step)
            writer.add_scalar("LearningRate/train", last_lr, global_step)
            writer.add_histogram("Model/final_conv.weight", model.final_conv.weight, global_step)

        running_loss += history["loss"][i]
        running_dice += history["dice"][i]
        running_hausdorff += history["hausdorff"][i]

        progress.set_postfix({
            "loss": f"{running_loss / (i + 1):.4f}",
            "dice": f"{running_dice / (i + 1):.4f}",
        })

    return history


def eval_step(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: callable,
        device: str,
        epoch: int = None,
        writer: SummaryWriter = None,
        test_set: bool = False,
) -> dict[str, np.ndarray]:
    running_loss = 0.0
    running_dice = 0.0
    running_hausdorff = 0.0
    n_batches = len(dataloader)
    history = {"loss": np.zeros(n_batches), "dice": np.zeros(n_batches), "hausdorff": np.zeros(n_batches)}

    desc = "Testing" if test_set else "Validation"
    progress = tqdm(enumerate(dataloader), total=n_batches, desc=desc, unit="batch")

    model.eval()
    with torch.no_grad():
        for i, (imgs, masks) in progress:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            history["loss"][i] = loss["combined"].item()
            history["dice"][i] = dice_score(preds, masks)
            history["hausdorff"][i] = hausdorff_distance(preds, masks)

            if not test_set and writer is not None:
                global_step = epoch * n_batches + i
                writer.add_scalar("Loss/val", history["loss"][i], global_step)
                writer.add_scalar("Dice/val", history["dice"][i], global_step)
                writer.add_scalar("Hausdorff/val", history["hausdorff"][i], global_step)

                if i == 0:
                    writer.add_images("Images/val", imgs, global_step)
                    writer.add_images("Masks/val", masks, global_step)
                    writer.add_images("Predictions/val", preds, global_step)

            running_loss += history["loss"][i]
            running_dice += history["dice"][i]
            running_hausdorff += history["hausdorff"][i]

            progress.set_postfix({
                "loss": f"{running_loss / (i + 1):.4f}",
                "dice": f"{running_dice / (i + 1):.4f}",
            })

    return history


def train():
    set_seed()
    print(f"Using device: {DEVICE}")

    model_name = "UNet_" + time.strftime("%d%m%Y-%H%M%S")
    log_dir = os.path.join("..", "logs", model_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logs at: {log_dir}")

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
    criterion = BCEDiceLoss()
    scaler = torch.amp.GradScaler()

    dummy_input = torch.rand((1, 3, *IMAGE_SIZE)).to(DEVICE)
    writer.add_graph(model, dummy_input)
    train_hist, val_hist = {}, {}

    for epoch in range(N_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{N_EPOCHS}]\n" + "-" * 30)

        train_hist = train_step(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            scaler,
            epoch,
            writer=writer,
            scheduler=scheduler
        )
        val_hist = eval_step(
            model,
            val_loader,
            criterion,
            DEVICE,
            epoch,
            writer=writer
        )

        train_loss = train_hist["loss"].mean()
        train_dice = train_hist["dice"].mean()
        val_loss = val_hist["loss"].mean()
        val_dice = val_hist["dice"].mean()

        print(f"lr: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Train Loss: {train_loss:.4f} - Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f}")

    print("-" * 30 + "\nTraining complete\n" + "-" * 30)

    test_hist = eval_step(model, test_loader, criterion, DEVICE, test_set=True)
    test_loss = test_hist["loss"].mean()
    test_dice = test_hist["dice"].mean()
    print(f"Test Loss: {test_loss:.4f} - Test Dice: {test_dice:.4f}")

    writer.add_graph(model, dummy_input)
    writer.add_hparams(
        {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'n_epochs': N_EPOCHS,
        },
        {
            'final_val_loss': val_hist["loss"][-1],
        }
    )

    model_path = os.path.join("..", "models", f"{model_name}_dice{int(test_dice * 100):}.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    train()
