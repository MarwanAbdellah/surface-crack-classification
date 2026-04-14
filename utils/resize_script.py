"""
Image resizing utility for the cracked/non-cracked dataset.

Reads raw images from the paths stored in a DataFrame, resizes each one to a
square target size using PIL, saves the result as a JPEG under a per-class
subdirectory, and returns the DataFrame extended with a 'resized_path' column.
All images are converted to grayscale ('L' mode) during the resize so the
saved files are single-channel and consistent with the rest of the pipeline.

Usage:
    from utils.resize_script import resize_images
    df = resize_images(df, resize_size=64, output_path='data/resized')
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd


def resize_images(df: pd.DataFrame, resize_size: int, output_path: str | Path) -> pd.DataFrame:
    """
    Resize images from a DataFrame and save them to an output directory.

    Args:
        df:          DataFrame with 'path' and 'class' columns.
        resize_size: Target width and height in pixels.
        output_path: Root directory where resized images will be saved.

    Returns:
        The input DataFrame with a new 'resized_path' column.
    """
    base_output_dir = Path(output_path)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    resized_paths = []

    print("Resizing images...")
    for i in tqdm(range(len(df))):
        original_path = Path(df.loc[i, 'path'])
        clas = df.loc[i, 'class']

        output_subdir = base_output_dir / clas
        output_subdir.mkdir(parents=True, exist_ok=True)

        new_filename = f"{original_path.stem}_resized{original_path.suffix}"
        new_path = output_subdir / new_filename

        img = Image.open(original_path).convert("L")
        img.resize((resize_size, resize_size)).save(new_path)

        resized_paths.append(str(new_path))

    df['resized_path'] = resized_paths
    return df