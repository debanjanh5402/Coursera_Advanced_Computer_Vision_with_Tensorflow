import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import tv_tensors

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split:str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        split = split.lower()
        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")
        
        self.is_train = (split == "train")

        # Load metadata
        images_df = pd.read_csv(os.path.join(root_dir, "images.txt"), sep=" ", header=None, names=["image_id", "image_path"])
        bboxes_df = pd.read_csv(os.path.join(root_dir, "bounding_boxes.txt"), sep=" ", header=None, names=["image_id", "x", "y", "width", "height"])
        labels_df = pd.read_csv(os.path.join(root_dir, "image_class_labels.txt"), sep=" ", header=None, names=["image_id", "class_id"])
        labels_df["class_id"] = labels_df["class_id"] - 1
        train_test_df = pd.read_csv(os.path.join(root_dir, "train_test_split.txt"), sep=" ", header=None, names=["image_id", "is_train"])
        df = images_df.merge(bboxes_df, on="image_id").merge(labels_df, on="image_id").merge(train_test_df, on="image_id")

        filter_value = 1 if self.is_train else 0
        self.df = df[df["is_train"] == filter_value].reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, "images", row["image_path"])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        xmin, ymin = row["x"], row["y"]
        xmax, ymax = xmin + row["width"], ymin + row["height"]
        bbox = tv_tensors.BoundingBoxes(data=[xmin, ymin, xmax, ymax], format="xyxy", canvas_size=(h, w), dtype=torch.float32)
        if self.transform: img, bbox = self.transform(img, bbox)
        new_h, new_w = bbox.canvas_size
        scale_tensor = torch.tensor([new_w, new_h, new_w, new_h], dtype=torch.float32)
        normalized_bbox = bbox.squeeze() / scale_tensor
        label = torch.tensor(row["class_id"], dtype=torch.int64)
        img_id = torch.tensor(row["image_id"], dtype=torch.int64)
        target = {"boxes": normalized_bbox, "labels": label, "image_id": img_id}
        return img, target
