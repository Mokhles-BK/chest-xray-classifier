"""Dataset + transforms for the NIH ChestX-ray14 multi-label task.

Reads decoded PNGs from `data/images/<split>/` and the multi-hot label vectors
from `data/manifest/<split>.csv`.  Images are resized to 224x224 and normalised
with ImageNet statistics, the standard for pretrained backbones.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGE_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def train_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(8),
            T.ColorJitter(brightness=0.05, contrast=0.05),
            T.ToTensor(),
            T.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def eval_transforms() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


class ChestXrayDataset(Dataset):
    """Dataset over a manifest CSV, decoding images on the fly from disk."""

    def __init__(self, manifest: pd.DataFrame, labels: list[str], images_dir: Path, train: bool) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.labels = labels
        self.images_dir = images_dir
        self.train = train
        self._target_cols = [f"label_{n}" for n in labels]
        self._transform = train_transforms() if train else eval_transforms()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        path = self.images_dir / row["filename"]
        img = Image.open(path).convert("RGB")
        target = torch.tensor([float(row[c]) for c in self._target_cols], dtype=torch.float32)
        return self._transform(img), target


def collate(samples: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([s[0] for s in samples])
    targets = torch.stack([s[1] for s in samples])
    return images, targets