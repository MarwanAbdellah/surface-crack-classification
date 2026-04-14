"""
Plotting utilities shared across all model notebooks.

Provides two functions:

  plot_training_curves  — side-by-side loss and accuracy curves with an
                          optional vertical marker at the best epoch.

  plot_confusion_matrix — heatmap of the sklearn confusion matrix with
                          class-name tick labels.

Usage:
    from utils.visualization import plot_training_curves, plot_confusion_matrix
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_curves(history: dict, title: str = ''):
    """
    Plot training & validation loss and accuracy curves side by side.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists.
        title:   Optional overall title (e.g. 'FFNN').
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'], label='Validation')
    if 'best_epoch' in history:
        ax1.axvline(x=history['best_epoch'], color='r', linestyle='--', alpha=0.5, label=f"Best (epoch {history['best_epoch']})")
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], label='Train')
    ax2.plot(epochs, history['val_acc'], label='Validation')
    if 'best_epoch' in history:
        ax2.axvline(x=history['best_epoch'], color='r', linestyle='--', alpha=0.5, label=f"Best (epoch {history['best_epoch']})")
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], title: str = 'Confusion Matrix'):
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true:      Ground truth labels.
        y_pred:      Predicted labels.
        class_names: List of class name strings.
        title:       Plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
