"""
Shared training and evaluation loop for all models in this project.

Provides two public functions:

  train_model  — runs the full training loop with early stopping,
                 optional LR scheduling, and best-checkpoint saving.

  evaluate_model — runs inference on a DataLoader and returns raw
                   predictions and ground-truth labels for downstream
                   metrics (classification report, confusion matrix).

Both functions accept an optional `forward_fn` argument so that models with
non-standard forward signatures (e.g. the RNN, which needs the input reshaped)
can plug in a custom call without modifying the shared loop.

Usage:
    from utils.training import train_model, evaluate_model
    history = train_model(model, train_loader, val_loader, criterion, optimizer)
    preds, labels = evaluate_model(model, test_loader)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


def train_model(
    model_or_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion=None,
    optimizer=None,
    epochs: int = 50,
    device: str | torch.device = 'cuda',
    scheduler=None,
    patience: int = 10,
    save_path: str | Path | None = None,
    forward_fn=None,
    verbose: bool = True,
    progress_bar: bool = True,
    params: dict | None = None,
    num_classes: int | None = None,
):
    """
    Shared training loop with early stopping and checkpointing.

    The first argument can be either:
      - An nn.Module (existing usage) — pass criterion and optimizer explicitly.
      - A model_fn callable: model_fn(params, num_classes) -> (model, optimizer, scheduler, criterion).
        Pass params= and num_classes= in this case; criterion/optimizer are ignored.

    Args:
        model_or_fn:  nn.Module to train, or a build function.
        train_loader: DataLoader for training set.
        val_loader:   DataLoader for validation set.
        criterion:    Loss function. Ignored when model_or_fn is a callable.
        optimizer:    Optimizer instance. Ignored when model_or_fn is a callable.
        epochs:       Maximum number of epochs.
        device:       Device to train on ('cuda' or 'cpu').
        scheduler:    Optional LR scheduler. Ignored when model_or_fn is a callable.
        patience:     Early stopping patience (epochs without val improvement).
        save_path:    Path to save the best model checkpoint. None = no saving.
        forward_fn:   Optional custom forward: forward_fn(model, inputs) -> logits.
        params:       Hyperparameter dict — required when model_or_fn is a callable.
        num_classes:  Number of output classes — required when model_or_fn is a callable.

    Returns:
        dict with keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc' (lists per epoch),
        and 'best_epoch' (int).
    """
    if callable(model_or_fn) and not isinstance(model_or_fn, nn.Module):
        model, optimizer, scheduler, criterion = model_or_fn(params, num_classes)
    else:
        model = model_or_fn

    device = torch.device(device)
    model = model.to(device)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
    }

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", disable=not progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = forward_fn(model, inputs) if forward_fn else model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", disable=not progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = forward_fn(model, inputs) if forward_fn else model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        # --- Metrics ---
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # --- LR Scheduler ---
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs} "
                f"| Train Loss: {epoch_train_loss:.4f} "
                f"| Train Acc: {epoch_train_acc:.4f} "
                f"| Val Loss: {epoch_val_loss:.4f} "
                f"| Val Acc: {epoch_val_acc:.4f} "
                f"| LR: {current_lr:.2e}",
                flush=True,
            )

        # --- Early Stopping & Checkpointing ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0

            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_acc': epoch_val_acc,
                }, save_path)
                if verbose:
                    print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    history['best_epoch'] = best_epoch
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str | torch.device = 'cuda',
    forward_fn=None,
):
    """
    Evaluate a trained model on a test set.

    Args:
        model:      The trained model.
        test_loader: DataLoader for the test set.
        device:     Device to run on.
        forward_fn: Optional custom forward (same as in train_model).

    Returns:
        tuple of (all_preds, all_labels) as lists of ints.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = forward_fn(model, inputs) if forward_fn else model(inputs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels
