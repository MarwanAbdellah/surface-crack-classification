"""
Generic Optuna hyperparameter search — works for FFNN, RNN, and CNN.

Usage in a notebook cell:

    from utils.hparam_search import run_search, plot_search_results
    from utils.config import FFNN_SEARCH_SPACE

    def build_ffnn(params, num_classes):
        model     = FFNN(num_classes=num_classes, dropout=params['dropout'])
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=params['lr'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=3)
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, scheduler, criterion

    best_params, trial_histories = run_search(
        build_ffnn,
        train_loader, val_loader,
        FFNN_SEARCH_SPACE,
        classes,
        results_path='Models/FFNN/best_hparams.json',
    )

    # Final training with best params
    model, optimizer, scheduler, criterion = build_ffnn(best_params, len(classes))
    history = train_model(model, train_loader, val_loader, criterion, optimizer, ...)

    plot_search_results(trial_histories, metric='val_acc', title='FFNN Optuna Search')
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, Subset


def _suggest(trial: optuna.Trial, name: str, spec: tuple):
    kind = spec[0]
    if kind == 'log_float':
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == 'float':
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == 'int':
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == 'categorical':
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Unknown search space kind: '{kind}'")


def _trial_label(trial: optuna.Trial) -> str:
    parts = [f"#{trial.number}"]
    for k, v in trial.params.items():
        parts.append(f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}")
    return ' '.join(parts)


def run_search(
    model_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    search_space: dict,
    classes: list,
    n_trials: int = 30,
    epochs: int = 50,
    patience: int = 5,
    train_frac: float = 0.3,
    results_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[dict, dict]:
    """
    Run an Optuna hyperparameter search and return best params + trial histories.

    Args:
        model_fn:      Callable(params, num_classes) -> (model, optimizer, scheduler, criterion).
        train_loader:  DataLoader for the training set.
        val_loader:    DataLoader for the validation set.
        search_space:  Dict of param_name -> spec from config.
        classes:       List of class names (used to derive num_classes).
        n_trials:      Number of Optuna trials.
        epochs:        Max epochs per trial (keep low — early stopping handles the rest).
        patience:      Early-stopping patience per trial.
        train_frac:    Fraction of training data used per search (default 0.3). CrackDataset keeps everything in RAM already, so this just subsets the indices.
        results_path:  If given, saves best params as JSON to this path.
        device:        Torch device. Defaults to CUDA if available.

    Returns:
        (best_params, trial_histories)
    """
    from utils.training import train_model

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = len(classes)
    trial_histories: dict = {}

    # Take a 30% random subset of training data for search — data is already in
    # RAM via CrackDataset's eager loading, so no extra copy is needed.
    train_n = int(len(train_loader.dataset) * train_frac)
    train_indices = torch.randperm(len(train_loader.dataset))[:train_n].tolist()
    search_train = Subset(train_loader.dataset, train_indices)

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
        model, optimizer, scheduler, criterion = model_fn(params, num_classes)

        batch_size = params.get('batch_size', train_loader.batch_size)
        subset_loader = DataLoader(search_train, batch_size=batch_size, shuffle=True, num_workers=0)
        val_subset_loader = DataLoader(val_loader.dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        print(f"\nTrial {trial.number + 1}/{n_trials} — " +
              ', '.join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in params.items()),
              flush=True)

        history = train_model(
            model, subset_loader, val_subset_loader,
            criterion, optimizer,
            epochs=epochs,
            device=device,
            scheduler=scheduler,
            patience=patience,
            save_path=None,
            verbose=True,
            progress_bar=True,
        )

        trial_histories[_trial_label(trial)] = history
        best_val = min(history['val_loss'])
        print(f"  -> best val_loss={best_val:.4f} over {len(history['val_loss'])} epochs", flush=True)
        torch.cuda.empty_cache()
        return best_val

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    print(f"\nBest trial #{best.number + 1}  val_loss={best.value:.4f}")
    print("Params:", best.params)

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(best.params, f, indent=2)
        print(f"Saved to {results_path}")

    return best.params, trial_histories


def plot_search_results(
    trial_histories: dict,
    metric: str = 'val_acc',
    title: str = '',
    top_n: int | None = None,
):
    """
    Plot a bar chart comparing Optuna trials using plot_hyperparameter_comparison.

    Args:
        trial_histories: The second return value of run_search.
        metric:          Metric to compare — 'val_acc', 'val_loss', etc.
        title:           Plot title.
        top_n:           If set, show only the top N trials by metric.
    """
    results = trial_histories
    if top_n is not None:
        reverse = 'loss' not in metric
        results = dict(
            sorted(
                trial_histories.items(),
                key=lambda kv: max(kv[1][metric]) if reverse else min(kv[1][metric]),
                reverse=reverse,
            )[:top_n]
        )

    names = list(results.keys())
    values = [
        min(h[metric]) if 'loss' in metric else max(h[metric])
        for h in results.values()
    ]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    plt.figure(figsize=(max(8, len(names) * 1.5), 5))
    bars = plt.bar(names, values, color=colors)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    plt.ylabel('Best ' + metric.replace('_', ' ').title())
    plt.title(title or f'Optuna Trials ({metric})')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
