"""
Eager-loading PyTorch Dataset for the cracked/non-cracked image classification task.

All images are loaded, transformed, and stored as tensors in RAM during __init__.
This removes per-epoch disk I/O — after the initial load, every epoch reads from
memory, which is significantly faster for repeated training runs (e.g. Optuna).

Note: online augmentation is not used. With eager loading, transforms are applied
once at load time, so random augmentations would produce the same augmented image
every epoch rather than a fresh one. Use offline augmentation if needed.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm


class CrackDataset(Dataset):
    """
    Eager-loading PyTorch Dataset for the cracked/non-cracked classification task.

    Loads all images from disk once during __init__, applies the transform pipeline,
    and stores the resulting tensors in RAM. After construction, all data access
    is memory-only — no disk reads during training.

    Args:
        df:        DataFrame with 'resized_path' and 'class' columns.
        transform: torchvision transform pipeline applied once to each image on load.
        classes:   Sorted list of class names. If None, derived from df['class'].
    """

    def __init__(self, df: pd.DataFrame, transform=None, classes: list[str] | None = None):
        if classes is None:
            classes = sorted(df['class'].unique())
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        labels = [self.class_to_idx[c] for c in df['class']]
        self.labels = torch.tensor(labels, dtype=torch.long)

        tensors = []
        for path in tqdm(df['resized_path'], desc='Loading dataset', leave=False):
            img = Image.open(path).convert('L')
            if transform:
                img = transform(img)
            tensors.append(img)

        self.data = torch.stack(tensors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
