import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from torch.utils.data import Dataset, DataLoader, Subset

from config import IDX2CLASS, STD, MEAN
from tqdm.auto import tqdm

def visualize_dataset(dataset, num_samples=16, cols=8, subplot_dim=4, title:str=None, save_dir:str=None):
    rows = math.ceil(num_samples / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*subplot_dim, rows*subplot_dim), dpi=300)
    axes = axes.flatten()

    for i in range(num_samples):
        img, targets = dataset[i]
        _, h, w = img.shape
        xmin, ymin, xmax, ymax = targets["boxes"]
        xmin, ymin, xmax, ymax = xmin*w, ymin*h, xmax*w, ymax*h
        img_id = targets["image_id"]
        class_id = targets["labels"]
        img = img * STD + MEAN  # Unnormalize
        ax = axes[i]
        ax.imshow(img.permute(1,2,0).numpy())
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Class {class_id.item()}: {IDX2CLASS[class_id.item()]}")
        ax.axis('off')
    # Hide any remaining axes if num_samples < rows*cols
    for j in range(num_samples, rows*cols):
        axes[j].axis('off')

    if title is not None: 
        plt.suptitle(title)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
    plt.show()


def plot_training_history(history:dict, save_dir:str=None):
    train_loss = history["train_loss"]; val_loss = history["val_loss"]
    train_cls_loss = history["train_cls_loss"]; val_cls_loss = history["val_cls_loss"]
    train_reg_loss = history["train_reg_loss"]; val_reg_loss = history["val_reg_loss"]
    train_acc = history["train_acc"]; val_acc = history["val_acc"]
    train_iou = history["train_iou"]; val_iou = history["val_iou"]

    epochs = range(1, len(train_loss)+1)

    plt.figure(1, figsize=(24, 6), dpi=300)
    plt.subplot(131)
    plt.plot(epochs, train_loss, "o-", label="Training")
    plt.plot(epochs, val_loss, "o-", label="Validation")     
    plt.title("Total Loss"); plt.legend(); plt.grid()

    plt.subplot(132)
    plt.plot(epochs, train_cls_loss, "o-", label="Training")
    plt.plot(epochs, val_cls_loss, "o-", label="Validation")     
    plt.title("Classification Loss"); plt.legend(); plt.grid()

    plt.subplot(133)
    plt.plot(epochs, train_reg_loss, "o-", label="Training")
    plt.plot(epochs, val_reg_loss, "o-", label="Validation")     
    plt.title("Bounding Box Regression Loss"); plt.legend(); plt.grid()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "training_losses.png"))
    plt.show()


    plt.figure(2, figsize=(16, 6), dpi=300)
    plt.subplot(121)
    plt.plot(epochs, train_acc, "o-", label="Training")
    plt.plot(epochs, val_acc, "o-", label="Validation")
    plt.title("Accuracy"); plt.legend(); plt.grid()

    plt.subplot(122)
    plt.plot(epochs, train_iou, "o-", label="Training")
    plt.plot(epochs, val_iou, "o-", label="Validation")
    plt.title("IoU"); plt.legend(); plt.grid()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.show()


def predict_and_visualize_on_dataset(dataset:Dataset, model:nn.Module, device:torch.device, 
                                     num_samples:int=16, cols:int=4, title:str=None, save_dir:str=None):
    indices = torch.randint(low=0, high=len(dataset), size=(num_samples,)).tolist()
    subset = Subset(dataset, indices)
    batch_size = min(len(subset), 32)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, prefetch_factor=None)
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []
    model.eval()
    with torch.inference_mode():
        for img_batch, _ in tqdm(dataloader, desc="Predicting", total=len(dataloader)):
            img_batch = img_batch.to(device)
            cls_out, bbox_out = model(img_batch)
            probs = torch.softmax(cls_out, dim=-1)
            pred_scores, pred_labels = torch.max(probs, dim=-1)
            all_pred_boxes.append(bbox_out.cpu())
            all_pred_labels.append(pred_labels.cpu())
            all_pred_scores.append(pred_scores.cpu())
            
    pred_boxes = torch.cat(all_pred_boxes, dim=0)
    pred_labels = torch.cat(all_pred_labels, dim=0)
    pred_scores = torch.cat(all_pred_scores, dim=0)

    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 6), dpi=300)
    axes = axes.flatten()

    for i in range(num_samples):
        img, targets = subset[i]
        _, h, w = img.shape
        img_unnorm = torch.clamp(img * STD + MEAN, 0, 1)
        ax = axes[i]
        ax.imshow(img_unnorm.permute(1, 2, 0).numpy())

        # Ground truth box
        gt_box = targets["boxes"]
        xmin, ymin, xmax, ymax = gt_box
        xmin, xmax = xmin * w, xmax * w
        ymin, ymax = ymin * h, ymax * h
        gt_rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor="#00FF00", facecolor='none')
        ax.add_patch(gt_rect)
        
        # Predicted Box
        p_box = pred_boxes[i]
        pxmin, pymin, pxmax, pymax = p_box
        pxmin, pxmax = pxmin * w, pxmax * w
        pymin, pymax = pymin * h, pymax * h
        pred_rect = patches.Rectangle((pxmin, pymin), pxmax - pxmin, pymax - pymin, linewidth=2, edgecolor='#FF0000', facecolor='none', linestyle='--')
        ax.add_patch(pred_rect)
        
        gt_label = targets["labels"]
        p_label = pred_labels[i]
        p_score = pred_scores[i]
        
        title_color = 'black' if gt_label == p_label else 'red'
        ax.set_title(f"GT: {IDX2CLASS[gt_label.item()]}\nPred: {IDX2CLASS[p_label.item()]}\nConf: {p_score:.3f}", color=title_color)
        ax.axis('off')

    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    handles = [patches.Patch(color='#00FF00', label='Ground Truth', fill=False, linewidth=2),
               patches.Patch(color='#FF0000', label='Prediction', fill=False, linewidth=2, linestyle='--')]
    fig.legend(handles=handles, loc='upper right', ncol=2, fontsize=12)
    if title is not None: fig.suptitle(title)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    if save_dir:
        fig.savefig(fname=os.path.join(save_dir, f"{title}.png"))
    plt.show()

    