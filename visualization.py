import numpy as np
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_maes, val_maes, title=None):
    # Plots training curves
    #
    # Args:
    # - train_losses: List of training losses per epoch
    # - val_losses: List of validation losses per epoch
    # - train_maes: List of training MAEs per epoch
    # - val_maes: List of validation MAEs per epoch
    # - title: Title of configuration
    # Returns:
    # *none* 

    epochs = np.arange(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if title is not None:
        fig.suptitle(title)

    # Left subplot: Loss
    axes[0].plot(epochs, train_losses, label="Training Loss")
    axes[0].plot(epochs, val_losses, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Over Epochs")
    axes[0].legend()
    axes[0].grid(True)

    # Right subplot: MAE
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