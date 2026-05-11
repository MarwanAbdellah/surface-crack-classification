# Surface Crack Classification

**Multi-Architecture Neural Network Comparison for Surface Fracture Detection**

> Binary image classification using FFNN, LSTM-RNN, CNN, and Transfer Learning (ResNet18)
> trained on a balanced dataset of ~228,000 grayscale surface images.

---

## Results at a Glance

| Model | Base Acc | Tuned Acc | Cracked F1 | Non-Cracked F1 |
|-------|:--------:|:---------:|:----------:|:--------------:|
| FFNN  | 70 %     | 74 %      | 0.67       | 0.78           |
| RNN (LSTM) | 73 % | 73 %    | 0.67       | 0.78           |
| CNN   | 80 %     | 80 %      | 0.78       | 0.83           |
| **Transfer Learning (ResNet18)** | 84 % | **86 %** | **0.85** | **0.86** |

> **Note on misclassification:** Across all from-scratch architectures, the **Cracked class is consistently harder to classify** than Non-Cracked. This is a direct consequence of microcracks — hairline fractures whose visual signature is nearly indistinguishable from normal surface texture. Models tend to predict Non-Cracked when in doubt, driving Cracked recall down (52 % FFNN → 54 % RNN → 68 % CNN) while Non-Cracked recall stays high. Transfer Learning achieves strong recall (81 % Cracked / 90 % Non-Cracked) by leveraging pretrained ImageNet features, improving from 84 % (baseline) to 86 % after hyperparameter tuning.

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
│   │   ├── FFNN.ipynb                # Feed-Forward NN (parametric depth, BatchNorm)
│   │   └── best_hparams.json
│   ├── RNN/
│   │   ├── RNN.ipynb                 # LSTM-based RNN (BatchNorm classifier)
│   │   └── best_hparams.json
│   ├── CNN/
│   │   ├── CNN.ipynb                 # Convolutional NN (BatchNorm + GAP head)
│   │   └── best_hparams.json
│   └── transfer_learning/
│       ├── TransferLearning.ipynb    # ResNet18 fine-tuned, 1-channel conv1
│       └── best_hparams.json
│
├── utils/
│   ├── dataset.py          # CrackDataset (eager-loading, transform-per-item)
│   ├── training.py         # train_model + evaluate_model
│   ├── hparam_search.py    # Optuna search + _make_optimizer / _make_scheduler helpers
│   ├── config.py           # Default configs + Optuna search spaces per model
│   ├── visualization.py    # Plotting helpers
│   ├── augmentation_script.py
│   └── resize_script.py
│
├── assets/                 # All figures used in the PDF report
│   ├── 4x4_view_matrix.png
│   ├── FFNN/
│   │   ├── FFNN_Base.png
│   │   ├── FFNN_CM.png
│   │   ├── FFNN_Hyperparameter_Search.png
│   │   ├── FFNN_Hyperparameter_Training.png
│   │   └── FFNN_Hyperparameter_CM.png
│   ├── CNN/
│   │   ├── CNN Architecture.png
│   │   ├── CNN_Base.png
│   │   ├── CNN_CM.png
│   │   ├── CNN_Hyperparameter_Search.png
│   │   ├── CNN_Hyperparameter_Training.png
│   │   └── CNN_Hyperparameter_CM.png
│   ├── RNN/
│   │   ├── RNN_Base.png
│   │   ├── RNN_CM.png
│   │   ├── RNN_Hyperparameter_Search.png
│   │   ├── RNN_Hyperparameter_Training.png
│   │   └── RNN_Hyperparameter_CM.png
│   └── Transfer_Learning/
│       ├── TL_Base_Training.png
│       ├── TL_Base_CM.png
│       ├── TL_Hyperparameter_Search.png
│       ├── TL_Hyperparameter_Training.png
│       └── TL_Hyperparameter_CM.png
│
├── PDF/
│   ├── main.tex            # LaTeX source for the full project report
│   └── main.pdf            # Compiled PDF
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
2. Build train (augmented) and eval (plain) datasets with 80/10/10 split
3. Preview augmented vs. eval samples (3×3 grids)
4. Define the model architecture
5. **Optuna hyperparameter search** (30 trials, 25 % data subset per trial)
6. **Final training** with best parameters on the full train set
7. Evaluate on the test set — classification report + confusion matrix

Saved checkpoints are written to `Models/saved_models/`.

---

## Utility Modules

### `utils/dataset.py` — `CrackDataset`
Eager-loading `torch.utils.data.Dataset`. All images are loaded once at construction
as PIL grayscale at source resolution (~1 GB RAM for 228k images at 64×64). The
transform pipeline is applied per `__getitem__` so per-epoch random augmentations
(`RandomHorizontalFlip`, `RandomVerticalFlip`) produce fresh variants every epoch.
Two instances are built per notebook — one with train augmentations, one without —
partitioned with the same `random_split` seed for a consistent 80/10/10 split.

### `utils/training.py` — `train_model` / `evaluate_model`
`train_model` is the shared training loop for all four architectures. Features:
- Early stopping (configurable patience, default 10)
- Automatic scheduler dispatch: `ReduceLROnPlateau` calls `step(val_loss)`, all others call `step()`
- Best-checkpoint saving via `torch.save`
- Accepts either a plain `nn.Module` or a factory `model_fn(params, num_classes)`

`evaluate_model` runs inference in `eval()` + `torch.no_grad()` mode and returns
flat prediction and label lists for downstream metrics.

### `utils/hparam_search.py` — `run_search`
Wraps an [Optuna](https://optuna.org/) study. Each trial uses a **25 %** random subset
of both training and validation data for speed. Returns the best parameter dict and
per-trial training histories for plotting. Two helpers are exported:
- `_make_optimizer(name, params, lr, weight_decay)` — dispatches Adam / SGD+momentum / RMSProp
- `_make_scheduler(name, optimizer, epochs)` — dispatches ReduceLROnPlateau / CosineAnnealingLR / StepLR

### `utils/config.py`
Default hyperparameter dictionaries (`FFNN_CONFIG`, `RNN_CONFIG`, `CNN_CONFIG`,
`TRANSFER_CONFIG`) and Optuna search-space definitions. All search spaces now include
`optimizer`, `scheduler`, and (for FFNN/CNN) `num_layers` to satisfy the full
project rubric for hyperparameter exploration.

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
