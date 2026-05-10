"""
Eager-loading PyTorch Dataset for the cracked/non-cracked image classification task.

All images are loaded once during __init__ as PIL grayscale at source resolution
and held in RAM (~1 GB for the full 228k dataset at 64x64). The transform pipeline
is applied per __getitem__ call, so random augmentations (RandomHorizontalFlip,
RandomVerticalFlip) produce a different sample every epoch.

With data already in RAM, use DataLoader with num_workers=0. Windows-spawn workers
add startup overhead rather than throughput for an in-memory dataset.
"""

from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    """
    Eager-loading PyTorch Dataset for the cracked/non-cracked classification task.

    Loads all images from disk once during __init__ as PIL grayscale images and
    stores them in RAM. After construction, all data access is memory-only.
    The transform is applied per __getitem__ so per-epoch random augmentations
    work correctly.

    Args:
        df:        DataFrame with 'resized_path' and 'class' columns.
        transform: torchvision transform pipeline applied to each image on access.
        classes:   Sorted list of class names. If None, derived from df['class'].
    """

    def __init__(self, df: pd.DataFrame, transform=None, classes: list[str] | None = None):
        if classes is None:
            classes = sorted(df['class'].unique())
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.labels = [self.class_to_idx[c] for c in df['class']]

        self.images = []
        for path in tqdm(df['resized_path'], desc='Loading dataset', leave=False):
            with Image.open(path) as img:
                pil = img.convert('L').copy()
            pil.load()  # materialise pixel data so the file handle can close
            self.images.append(pil)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
