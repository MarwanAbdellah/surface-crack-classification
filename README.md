# Surface Crack Classification

**Multi-Architecture Neural Network Comparison for Surface Fracture Detection**

> Binary image classification using FFNN, LSTM-RNN, and CNN architectures trained on a
> balanced dataset of ~228,000 grayscale surface images.

---

## Results at a Glance

| Model | Test Accuracy | Cracked F1 | Non-Cracked F1 | Val Loss |
|-------|:-------------:|:----------:|:--------------:|:--------:|
| FFNN  | 71 %          | 0.63       | 0.76           | 0.5754   |
| RNN (LSTM) | 73 %     | 0.67       | 0.78           | 0.4849   |
| **CNN** | **80 %**  | **0.77**   | **0.81**       | **0.3725** |

---

## Dataset

Source: [Cracked and Non-Cracked Surface Datasets](https://www.kaggle.com/datasets/geadalfa/cracked-non-cracked-surface-datasets/data) — Kaggle

| Stage | Details |
|-------|---------|
| Raw images | Mixed 256×256 and 227×227 |
| After resizing | 227×227 uniform |
| After augmentation (×3) | ~304,000 images |
| After class balancing | **227,872 images** |
| Train / Val / Test split | 80 % / 10 % / 10 % (seed 42) |

---

## Project Structure

```
nn/
├── Notebooks/
│   ├── 1.Data_Warehouse.ipynb        # Raw data inventory → images_path.csv
│   ├── 2.Data_Visualization.ipynb    # EDA, size discovery, 4×4 sample grid
│   ├── 3.Images_Preprocessing.ipynb  # Uniform resize to 227×227
│   ├── 4.Image_augmentation.ipynb    # Flip + ColorJitter augmentation (×3)
│   └── 5.Images_Imbalance.ipynb      # Majority undersampling → trainable_df.csv
│
├── Models/
│   ├── FFNN/
│   │   ├── FFNN.ipynb                # Feed-Forward NN
│   │   └── best_hparams.json
│   ├── RNN/
│   │   ├── RNN.ipynb                 # LSTM-based RNN
│   │   └── best_hparams.json
│   └── CNN/
│       ├── CNN.ipynb                 # Convolutional NN
│       └── best_hparams.json
│
├── utils/
│   ├── dataset.py          # CrackDataset (lazy-loading PyTorch Dataset)
│   ├── training.py         # train_model + evaluate_model
│   ├── hparam_search.py    # Optuna-based hyperparameter search
│   ├── config.py           # Default hyperparameter configs per model
│   ├── visualization.py    # Plotting helpers
│   ├── augmentation_script.py
│   └── resize_script.py
│
├── assets/                 # All figures used in the PDF report
│   ├── 4x4_view_matrix.png
│   ├── FFNN/
│   ├── RNN/
│   └── CNN/
│
├── data/                   # (not tracked) downloaded dataset + processed CSVs
├── pyproject.toml
└── uv.lock
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management
(Python ≥ 3.13).

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install project dependencies
uv sync
```

**PyTorch is not listed in `pyproject.toml`** because it must be installed separately
with the wheel that matches your CUDA version. Use `uv pip install` directly:

```bash
# Example: CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Example: CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU-only
uv pip install torch torchvision torchaudio
```

---

## Data Preparation

Download the dataset from Kaggle and place it at `data/Bangunan Retak/`.
Then run the notebooks **in order**:

```
1.Data_Warehouse.ipynb        →  data/images_path.csv
2.Data_Visualization.ipynb    →  (exploration only)
3.Images_Preprocessing.ipynb  →  data/df_resized.csv
4.Image_augmentation.ipynb    →  data/df_augmented.csv
5.Images_Imbalance.ipynb      →  data/trainable_df.csv  ← used by all models
```

---

## Running the Models

Open each model notebook inside `Models/` and run all cells.
Each notebook follows the same structure:

1. Load `data/trainable_df.csv`
2. Build the dataset and 80/10/10 split
3. Define the model architecture
4. **Baseline training** with default config from `utils/config.py`
5. **Optuna hyperparameter search** (30 trials, 30 % data subset per trial)
6. **Final training** with best parameters
7. Evaluate on the test set — classification report + confusion matrix

Saved checkpoints are written to `Models/saved_models/`.

---

## Utility Modules

### `utils/dataset.py` — `CrackDataset`
Lazy-loading `torch.utils.data.Dataset`. Stores file paths and labels at construction
time; images are opened and transformed on demand in `__getitem__`. Works seamlessly
with `random_split` and `DataLoader` workers.

### `utils/training.py` — `train_model` / `evaluate_model`
`train_model` is the shared training loop for all three architectures. Features:
- Early stopping (default patience = 10)
- `ReduceLROnPlateau` learning-rate scheduling
- Best-checkpoint saving via `torch.save`
- Accepts either a plain `nn.Module` or a factory `model_fn(params, num_classes)`

`evaluate_model` runs inference in `eval()` + `torch.no_grad()` mode and returns
flat prediction and label lists for downstream metrics.

### `utils/hparam_search.py` — `run_search`
Wraps an [Optuna](https://optuna.org/) study. Each trial uses a 30 % random subset
of the training data for speed. Returns the best parameter dict and per-trial
training histories for plotting.

### `utils/config.py`
Default hyperparameter dictionaries (`FFNN_CONFIG`, `RNN_CONFIG`, `CNN_CONFIG`) and
Optuna search-space definitions (`FFNN_SEARCH_SPACE`, …).

---

## Dependencies

| Package | Managed by | Purpose |
|---------|-----------|---------|
| `torch` / `torchvision` | `uv pip install` (GPU wheel) | Model building & training |
| `pandas` | `uv sync` | DataFrame-based data pipeline |
| `numpy` | `uv sync` | Numerical operations |
| `Pillow` / `opencv-python` | `uv sync` | Image loading & preprocessing |
| `optuna` | `uv sync` | Hyperparameter search |
| `scikit-learn` | `uv sync` | Classification report, confusion matrix |
| `matplotlib` / `seaborn` | `uv sync` | Visualisation |
| `tqdm` | `uv sync` | Progress bars |
| `ipykernel` | `uv sync` | Jupyter notebook support |
