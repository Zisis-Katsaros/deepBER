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
        fig.suptitle(title+": Training Curves", fontsize=16)

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

def plot_predicted_vs_actual(true_labels, predictions, title=None):
    # Plots predicted values against the corresponding actual values
    #
    # Args:
    # - true_labels: True target values
    # - predictions: Model predictions
    # - title: Optional plot title
    # Returns:
    # *none*

    true_labels = np.asarray(true_labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    if true_labels.shape != predictions.shape:
        raise ValueError("true_labels and predictions must have the same shape.")

    fig, ax = plt.subplots(figsize=(6, 6))

    if title is not None:
        ax.set_title(title+": Predicted vs. Actual", fontsize=16)
    else:
        ax.set_title("Predicted vs. Actual", fontsize=16)

    ax.scatter(true_labels, predictions, alpha=0.7)

    min_value = min(true_labels.min(), predictions.min())
    max_value = max(true_labels.max(), predictions.max())
    ax.plot([min_value, max_value], [min_value, max_value], "r--", label="Ideal")

    ax.set_xlabel("True Labels")
    ax.set_ylabel("Model Prediction")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals(true_labels, predictions, title=None):
    # Plots residuals against the corresponding true values
    #
    # Args:
    # - true_labels: True target values
    # - predictions: Model predictions
    # - title: Optional plot title
    # Returns:
    # *none*

    true_labels = np.asarray(true_labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    if true_labels.shape != predictions.shape:
        raise ValueError("true_labels and predictions must have the same shape.")

    residuals = true_labels - predictions

    fig, ax = plt.subplots(figsize=(6, 6))

    if title is not None:
        ax.set_title(title+": Residual Plot", fontsize=16)
    else:
        ax.set_title("Residual Plot", fontsize=16)

    ax.scatter(true_labels, residuals, alpha=0.7)
    ax.axhline(0, color="r", linestyle="--", label="Zero Residual")

    ax.set_xlabel("True Labels")
    ax.set_ylabel("Residuals")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(true_labels, predictions, bins=30, title=None):
    # Plots the histogram of residuals to show the error distribution
    #
    # Args:
    # - true_labels: True target values
    # - predictions: Model predictions
    # - bins: Number of histogram bins
    # - title: Optional plot title
    # Returns:
    # *none*

    true_labels = np.asarray(true_labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    if true_labels.shape != predictions.shape:
        raise ValueError("true_labels and predictions must have the same shape.")

    residuals = true_labels - predictions

    fig, ax = plt.subplots(figsize=(6, 4))

    if title is not None:
        ax.set_title(title+": Error Distribution", fontsize=16)
    else:
        ax.set_title("Error Distribution", fontsize=16)

    ax.hist(residuals, bins=bins, edgecolor="black", alpha=0.75)
    ax.axvline(0, color="r", linestyle="--", label="Zero Error")

    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_error_vs_feature(feature, true_labels, predictions, title=None, feature_name=""):
    # Plots prediction error against a feature's values
    #
    # Args:
    # - feature: Feature values for the x-axis
    # - true_labels: True target values
    # - predictions: Model predictions
    # - title: Optional plot title
    # Returns:
    # *none*

    feature = np.asarray(feature).reshape(-1)
    true_labels = np.asarray(true_labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    if feature.shape != true_labels.shape or true_labels.shape != predictions.shape:
        raise ValueError("feature, true_labels, and predictions must have the same shape.")

    error = true_labels - predictions

    fig, ax = plt.subplots(figsize=(6, 6))

    if title is not None:
        ax.set_title(title+": Error vs. Feature - "+feature_name, fontsize=16)
    else:
        ax.set_title("Error vs. Feature", fontsize=16)

    ax.scatter(feature, error, alpha=0.7)
    ax.axhline(0, color="r", linestyle="--", label="Zero Error")

    ax.set_xlabel("Feature Values")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()