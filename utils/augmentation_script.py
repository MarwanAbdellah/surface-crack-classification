"""
Offline image augmentation utility for the cracked/non-cracked dataset.

Generates augmented copies of every image in the input DataFrame and saves
them to disk, then returns a combined DataFrame of original + augmented rows.
This script is used for *offline* (pre-training) data augmentation — it
physically writes new image files. For *online* augmentation applied on-the-fly
during training see the per-model transform pipelines in each model notebook.

All augmentation is performed on grayscale images (PIL mode 'L'). The caller
supplies any torchvision-compatible transform callable as the `augmentation`
argument; only transforms that preserve grayscale (spatial ops such as flips,
rotations, crops) are appropriate here — colour transforms will fail on 'L'
mode images.

Usage:
    from utils.augmentation_script import augment_images
    df_aug = augment_images(df, output_path='data/augmented',
                            augmentation=my_transform, num_aug_per_image=3)
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd

def augment_images(df, output_path, augmentation, num_aug_per_image=3, **kwargs):
    aug_output_dir = Path(output_path)
    aug_output_dir.mkdir(parents=True, exist_ok=True)

    augmented_rows = []

    print("Augmenting images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_path = row['resized_path']
        img = Image.open(original_path).convert("L")

        for i in range(num_aug_per_image):
            aug_img = augmentation(img)
            new_filename = f"{Path(original_path).stem}_aug{i}.jpeg"
            new_path = aug_output_dir / new_filename
            aug_img.save(new_path)

            augmented_rows.append({
                'resized_path': str(new_path),
                'class': row['class']
            })

    df_augmented = pd.DataFrame(augmented_rows)
    return pd.concat([df, df_augmented], ignore_index=True)