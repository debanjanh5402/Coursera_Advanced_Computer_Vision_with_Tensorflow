import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from engine import compute_iou_pixel

from config import TEST_TRANSFORM, MEAN, STD, IDX2CLASS
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def predict_on_dataset(dataset: Dataset, model: nn.Module, device: torch.device):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_bboxes = []; all_labels = []; all_scores = []
    
    model.eval()
    
    with torch.inference_mode():
        for img_batch, _ in tqdm(dataloader, desc="Predicting", total=len(dataloader)):
            img_batch = img_batch.to(device)
            cls_out, bbox_out = model(img_batch)
            probs = torch.softmax(cls_out, dim=-1)
            scores, labels = torch.max(probs, dim=-1)
            all_bboxes.append(bbox_out.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())
    final_bboxes = torch.cat(all_bboxes, dim=0)   # Shape: (N, 4)
    final_labels = torch.cat(all_labels, dim=0)   # Shape: (N,)
    final_scores = torch.cat(all_scores, dim=0)   # Shape: (N,)
    
    return final_bboxes, final_labels, final_scores



def predict(path:str, model:nn.Module, device:torch.device):
    raw_img = Image.open(path).convert("RGB")
    img_tensor = TEST_TRANSFORM(raw_img).unsqueeze(dim=0).to(device)

    model.eval()
    with torch.inference_mode():
        cls_out, bbox_out = model(img_tensor)
        probs = torch.softmax(cls_out, dim=-1)
        pred_score, pred_label = torch.max(probs, dim=-1)
        p_box = bbox_out[0].cpu()
        p_label_idx = pred_label.item()
        p_score = pred_score.item()

    img = img_tensor.squeeze(dim=0)
    img = torch.clamp(img*STD+MEAN, 0, 1)
    _, h, w = img.shape

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.imshow(img.permute(1, 2, 0).numpy())
    
    # Unnormalize bounding box from [0, 1] relative coordinates to absolute pixels
    pxmin, pymin, pxmax, pymax = p_box
    pxmin, pxmax = pxmin * w, pxmax * w
    pymin, pymax = pymin * h, pymax * h
    
    rect = patches.Rectangle((pxmin, pymin), pxmax - pxmin, pymax - pymin, 
                             linewidth=2, edgecolor='#FF0000', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    class_name = IDX2CLASS.get(p_label_idx, f"Unknown ID: {p_label_idx}")
    ax.set_title(f"Pred: {class_name}\nConf: {p_score:.3f}", fontsize=12, color='black')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def results_on_dataset(dataset: Dataset, model: nn.Module, device: torch.device):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    gt_bboxes = []; gt_labels = []
    pred_bboxes = []; pred_labels = []; pred_scores = []
    
    model.eval()
    
    with torch.inference_mode():
        for img_batch, target_batch in tqdm(dataloader, desc="Predicting", total=len(dataloader)):
            img_batch = img_batch.to(device)
            gt_labels.append(target_batch["labels"].cpu())
            gt_bboxes.append(target_batch["boxes"].cpu())

            cls_out, bbox_out = model(img_batch)
            probs = torch.softmax(cls_out, dim=-1)
            scores, labels = torch.max(probs, dim=-1)
            pred_bboxes.append(bbox_out.cpu())
            pred_labels.append(labels.cpu())
            pred_scores.append(scores.cpu())

    pred_bboxes = torch.cat(pred_bboxes, dim=0)   # Shape: (N, 4)
    pred_labels = torch.cat(pred_labels, dim=0)   # Shape: (N,)
    pred_scores = torch.cat(pred_scores, dim=0)   # Shape: (N,)
    gt_bboxes = torch.cat(gt_bboxes, dim=0)
    gt_labels = torch.cat(gt_labels, dim=0)

    
    return {"gt_bboxes": gt_bboxes,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "pred_bboxes": pred_bboxes,
            "pred_scores": pred_scores}


def evaluate_results(results_dict, image_size=(224, 224)):
    """
    results_dict: output of results_on_dataset()
    image_size: (h, w)
    """

    gt_labels = results_dict["gt_labels"].numpy()
    pred_labels = results_dict["pred_labels"].numpy()
    gt_bboxes = results_dict["gt_bboxes"]
    pred_bboxes = results_dict["pred_bboxes"]

    h, w = image_size

    # ===============================
    # Classification Metrics
    # ===============================
    acc = accuracy_score(gt_labels, pred_labels)
    precision_macro = precision_score(gt_labels, pred_labels, average="macro", zero_division=0)
    recall_macro = recall_score(gt_labels, pred_labels, average="macro", zero_division=0)
    f1_macro = f1_score(gt_labels, pred_labels, average="macro", zero_division=0)

    precision_weighted = precision_score(gt_labels, pred_labels, average="weighted", zero_division=0)
    recall_weighted = recall_score(gt_labels, pred_labels, average="weighted", zero_division=0)
    f1_weighted = f1_score(gt_labels, pred_labels, average="weighted", zero_division=0)

    print("="*60)
    print("Classification Metrics")
    print("="*60)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(gt_labels, pred_labels, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(gt_labels, pred_labels)
    print("\nConfusion Matrix shape:", cm.shape)

    # ===============================
    # Bounding Box IoU Metrics
    # ===============================
    ious = compute_iou_pixel(pred_bboxes, gt_bboxes, h, w)
    ious = ious.cpu().numpy()

    mean_iou = np.mean(ious)
    std_iou = np.std(ious)

    print("="*60)
    print("Bounding Box Metrics")
    print("="*60)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Std IoU: {std_iou:.4f}")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "confusion_matrix": cm,
        "ious": ious
    }

def plot_iou_distribution(ious, mean_iou, std_iou):
    plt.figure(figsize=(9,6), dpi=300)
    plt.scatter(range(1, len(ious)+1), ious)
    plt.title("IoU Distribution")
    plt.axhline(y=mean_iou, color="green", linestyle="-")
    plt.axhline(y=mean_iou+3*std_iou, color="red", linestyle="--")
    plt.axhline(y=mean_iou-3*std_iou, color="red", linestyle="--")
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10,10), dpi=300)
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.show()



