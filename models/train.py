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
from models.deeplab_v3p import DeepLabV3Plus
from models.unet import UNet
from utils.constants import (
    N_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    set_seed
)
from utils.metrics import BCEDiceLoss, update_history


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

        if writer is not None:
            global_step = epoch * n_batches + i
            writer.add_scalar("Train/Loss", history["loss"][i], global_step)
            writer.add_scalar("Train/Dice", history["dice"][i], global_step)
            writer.add_scalar("Train/Hausdorff", history["hausdorff"][i], global_step)

            if scheduler is not None:
                last_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Train/LearningRate", last_lr, global_step)

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
        writer: SummaryWriter = None,
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

            if not test_set and writer is not None:
                global_step = epoch * n_batches + i
                writer.add_scalar("Val/Loss", history["loss"][i], global_step)
                writer.add_scalar("Val/Dice", history["dice"][i], global_step)
                writer.add_scalar("Val/Hausdorff", history["hausdorff"][i], global_step)

                if i == 0:
                    writer.add_images("Val/Images", imgs, global_step)
                    writer.add_images("Val/Masks", masks, global_step)
                    writer.add_images("Val/Predictions", preds, global_step)

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

    log_dir = os.path.join("..", "logs", model_name_ext)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logs at: {log_dir}")

    train_set = ImageDataset("../data/train", transforms=get_train_transforms())
    val_set = ImageDataset("../data/val", transforms=get_val_transforms())
    test_set = ImageDataset("../data/test", transforms=get_val_transforms())

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
            writer=writer,
        )
        val_hist = eval_step(
            model,
            val_loader,
            criterion,
            DEVICE,
            epoch,
            writer=writer
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
            best_model_path = os.path.join("..", "bin", f"{model_name_ext}.pth")

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

    # For some reason, the metrics dict are not being shown in tensorboard,
    # so we moved them to the hparams dict.
    writer.add_hparams(
        {
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
        },
        {}
    )

    writer.flush()
    writer.close()
    print()


if __name__ == "__main__":
    train(UNet(3, 1))
    train(DeepLabV3Plus(3, 1))
