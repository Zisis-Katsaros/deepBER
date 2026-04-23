import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_maes, val_maes, title=None):
    epochs = np.arange(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if title is not None:
        fig.suptitle(title)

    axes[0].plot(epochs, train_losses, label="Training Loss")
    axes[0].plot(epochs, val_losses, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Over Epochs")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_maes, label="Training MAE")
    axes[1].plot(epochs, val_maes, label="Validation MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE Over Epochs")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if title is not None:
        plt.subplots_adjust(top=0.85)
    plt.show()