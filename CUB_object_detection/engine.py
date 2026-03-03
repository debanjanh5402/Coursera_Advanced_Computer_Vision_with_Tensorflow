import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
from tqdm.auto import tqdm
from typing import Dict
import numpy as np


# ================================
# IoU in Pixel Space
# ================================
def compute_iou_pixel(box1, box2, h, w):
    """
    box1, box2: (B, 4) normalized xyxy
    h, w: image height and width (int)
    """

    # Convert normalized → pixel space
    scale = torch.tensor([w, h, w, h], device=box1.device)
    box1 = box1 * scale
    box2 = box2 * scale

    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1 + area2 - inter + 1e-6

    iou = inter / union
    return iou


# ================================
# Training Step
# ================================
def train_step(model: nn.Module,
               dataloader: DataLoader,
               optimizer: Optimizer,
               loss_functions: Dict[str, nn.Module],
               loss_weights: Dict[str, float],
               device: torch.device):

    model.train()

    total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
    total_iou = 0.0
    total_correct, total_samples = 0, 0

    cls_loss_fn = loss_functions["classification"]
    reg_loss_fn = loss_functions["regression"]

    cls_weight = loss_weights["classification"]
    reg_weight = loss_weights["regression"]

    for img_batch, target_batch in tqdm(dataloader, desc="Training", leave=False):

        img_batch = img_batch.to(device)
        label_batch = target_batch["labels"].to(device)
        bbox_batch = target_batch["boxes"].to(device)

        batch_size, _, h, w = img_batch.shape

        optimizer.zero_grad()

        cls_out, bbox_out = model(img_batch)

        # Loss
        cls_loss = cls_loss_fn(cls_out, label_batch)
        reg_loss = reg_loss_fn(bbox_out, bbox_batch)
        loss = cls_weight * cls_loss + reg_weight * reg_loss

        loss.backward()
        optimizer.step()

        # Accuracy
        preds = torch.argmax(cls_out, dim=1)
        total_correct += (preds == label_batch).sum().item()

        # IoU (pixel space)
        ious = compute_iou_pixel(bbox_out, bbox_batch, h, w)
        total_iou += ious.sum().item()

        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_reg_loss += reg_loss.item() * batch_size
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_cls_loss / total_samples,
        total_reg_loss / total_samples,
        total_correct / total_samples,
        total_iou / total_samples
    )


# ================================
# Validation Step
# ================================
def validation_step(model: nn.Module,
                    dataloader: DataLoader,
                    loss_functions: Dict[str, nn.Module],
                    loss_weights: Dict[str, float],
                    device: torch.device):

    model.eval()

    total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
    total_iou = 0.0
    total_correct, total_samples = 0, 0

    cls_loss_fn = loss_functions["classification"]
    reg_loss_fn = loss_functions["regression"]

    cls_weight = loss_weights["classification"]
    reg_weight = loss_weights["regression"]

    with torch.inference_mode():

        for img_batch, target_batch in tqdm(dataloader, desc="Validation", leave=False):

            img_batch = img_batch.to(device)
            label_batch = target_batch["labels"].to(device)
            bbox_batch = target_batch["boxes"].to(device)

            batch_size, _, h, w = img_batch.shape

            cls_out, bbox_out = model(img_batch)

            cls_loss = cls_loss_fn(cls_out, label_batch)
            reg_loss = reg_loss_fn(bbox_out, bbox_batch)
            loss = cls_weight * cls_loss + reg_weight * reg_loss

            # Accuracy
            preds = torch.argmax(cls_out, dim=1)
            total_correct += (preds == label_batch).sum().item()

            # IoU
            ious = compute_iou_pixel(bbox_out, bbox_batch, h, w)
            total_iou += ious.sum().item()

            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            total_reg_loss += reg_loss.item() * batch_size
            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_cls_loss / total_samples,
        total_reg_loss / total_samples,
        total_correct / total_samples,
        total_iou / total_samples
    )


# ================================
# Model Fit
# ================================
def model_fit(model: nn.Module,
              train_dataloader: DataLoader,
              test_dataloader: DataLoader,
              optimizer: Optimizer,
              loss_functions: Dict[str, nn.Module],
              loss_weights: Dict[str, float],
              device: torch.device,
              epochs: int,
              save_dir: str,
              resume:bool, 
              best_or_latest:str,
              scheduler=None):

    history = {
        "train_loss": [], "train_cls_loss": [], "train_reg_loss": [],
        "train_acc": [], "train_iou": [],
        "val_loss": [], "val_cls_loss": [], "val_reg_loss": [],
        "val_acc": [], "val_iou": []
    }

    best_val_loss = np.inf
    start_epoch = 0

    if resume:
        checkpoint_path = os.path.join(save_dir, f"{best_or_latest}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
            best_val_loss = checkpoint["best_val_loss"]

            print(f"[INFO] Resume training from epoch {start_epoch}. Best Validation Loss: {best_val_loss:0.6f}")

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", initial=start_epoch, total=epochs):

        train_loss, train_cls_loss, train_reg_loss, train_acc, train_iou = train_step(model, train_dataloader, optimizer, 
                                                                                      loss_functions, loss_weights, device)

        val_loss, val_cls_loss, val_reg_loss, val_acc, val_iou = validation_step(model, test_dataloader, loss_functions, 
                                                                                 loss_weights, device)

        history["train_loss"].append(train_loss)
        history["train_cls_loss"].append(train_cls_loss)
        history["train_reg_loss"].append(train_reg_loss)
        history["train_acc"].append(train_acc)
        history["train_iou"].append(train_iou)

        history["val_loss"].append(val_loss)
        history["val_cls_loss"].append(val_cls_loss)
        history["val_reg_loss"].append(val_reg_loss)
        history["val_acc"].append(val_acc)
        history["val_iou"].append(val_iou)

        print("-"*51, f"    Epoch [{epoch+1}/{epochs}] Summary  ", "-"*51)
        print(f"||     TRAINING STEP     || total loss: {train_loss:0.4f}, classification loss: {train_cls_loss:0.4f}, regression loss: {train_reg_loss:0.4f}, accuracy: {train_acc:0.4f}, iou: {train_iou:0.4f}")
        print(f"||    VALIDATION STEP    || total loss: {val_loss:0.4f}, classification loss: {val_cls_loss:0.4f}, regression loss: {val_reg_loss:0.4f}, accuracy: {val_acc:0.4f}, iou: {val_iou:0.4f}")

        is_best = val_loss < best_val_loss

        if is_best:
            best_path = os.path.join(save_dir, "best_checkpoint.pth")
            print(f"[INFO] Validation Loss improved from {best_val_loss:0.6f} to {val_loss:0.6f}. Saving checkpoint at: {best_path}")
            best_val_loss = val_loss

        checkpoint = {"epoch": epoch + 1,
                      "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "best_val_loss": best_val_loss,
                      "history": history
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, "best_checkpoint.pth"))

        torch.save(checkpoint, os.path.join(save_dir, "latest_checkpoint.pth"))

        # Step LR scheduler (once per epoch)
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[INFO] Learning Rate updated to: {current_lr:.6f}")
        
    return history