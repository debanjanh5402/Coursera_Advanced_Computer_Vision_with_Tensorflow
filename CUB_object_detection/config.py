import os
import random
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TRAIN_TRANSFORM = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
    v2.RandomRotation(10),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=MEAN, std=STD),
])

TEST_TRANSFORM = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=MEAN, std=STD),
])

df = pd.read_csv(filepath_or_buffer="/Users/debanjan_5402/Desktop/Dataset/CUB_200_2011/classes.txt", 
                 sep=" ", header=None, names=["id", "class_name"])
df["id"] = df["id"] - 1
df["class_name"] = df["class_name"].str.split(".").str[1]
IDX2CLASS = dict(zip(df["id"], df["class_name"]))
CLASS2IDX = dict(zip(df["class_name"], df["id"]))


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[INFO] Global seed set to {seed}")