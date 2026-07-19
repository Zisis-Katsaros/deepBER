import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

def plot_s_param_pred_vs_act_from_pistcnn(test_targets: np.ndarray, test_preds: np.ndarray, freq_array: np.ndarray, 
                          port_i: int, port_j: int, geom_idx: int = 0, num_ports: int = 18, 
                          save_dir: str = "plots", close_figure: bool = True):
    p1, p2 = min(port_i, port_j), max(port_i, port_j)
    
    idx = 0
    channel_map = {}
    for i in range(1, num_ports + 1):
        for j in range(i, num_ports + 1):
            channel_map[(i, j)] = idx
            idx += 1
            
    target_idx = channel_map[(p1, p2)]
    num_channels = len(channel_map) # 171 unique elements for 18 ports
    
    # Extract Real and Imaginary 1D arrays for the specific geometry
    # Real parts are in the first half of the channels, Imaginary parts in the second half
    real_idx = target_idx
    imag_idx = target_idx + num_channels
    
    actual_real = test_targets[geom_idx, real_idx, :]
    pred_real = test_preds[geom_idx, real_idx, :]
    
    actual_imag = test_targets[geom_idx, imag_idx, :]
    pred_imag = test_preds[geom_idx, imag_idx, :]
    
    # Call your custom plotting function for both Real and Imaginary components
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot Real Part
    plot_preds_vs_act_freq(
        true_labels=actual_real,
        predictions=pred_real,
        freq_array=freq_array,
        title=f"Sample {geom_idx+1} - Re(S{p1}{p2})",
        save_path=os.path.join(save_dir, f"sample_{geom_idx+1}_Re_S{p1}_{p2}.png"),
        close_figure=close_figure
    )  
    # Plot Imaginary Part
    plot_preds_vs_act_freq(
        true_labels=actual_imag,
        predictions=pred_imag,
        freq_array=freq_array,
        title=f"Sample {geom_idx+1} - Im(S{p1}{p2})",
        save_path=os.path.join(save_dir, f"sample_{geom_idx+1}_Im_S{p1}_{p2}.png"),
        close_figure=close_figure
    )


def  plot_abcd_preds_vs_act_freq(a_labels_array, a_preds_array, b_labels_array, b_preds_array, c_labels_array, c_preds_array, d_labels_array, d_preds_array, 
                                freq_array, title=None, save_path=None, close_figure: bool =True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax_idx, (labels_array, preds_array, prefix) in enumerate(zip(
        [a_labels_array, b_labels_array, c_labels_array, d_labels_array],
        [a_preds_array, b_preds_array, c_preds_array, d_preds_array],
        ["A", "B", "C", "D"]
    )):
        ax = axes[ax_idx // 2, ax_idx % 2]
        ax.scatter(freq_array, labels_array, color='blue', alpha=0.7, label=f'{prefix}-Actual')
        ax.scatter(freq_array, preds_array, color='orange', alpha=0.7, label=f'{prefix}-Predicted')

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(f"{prefix} Predictions/ Actuals")
        ax.legend()
        ax.grid(True) 

    plt.tight_layout()
    if title is not None:
        plt.subplots_adjust(top=0.85)

    # Saving plot
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {save_path}")

    if not close_figure:
        plt.show()
    if close_figure:
        plt.close(fig)
         

def plot_pki_vs_act_freq(true_labels, pki, freq_array, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if title is not None:
        ax.set_title(title + ": PKI vs. Actual", fontsize=16)
    else:
        ax.set_title("PKI vs. Actual", fontsize=16)

    ax.scatter(freq_array, true_labels, color='blue', alpha=0.7, label='Actual Waveform')
    ax.scatter(freq_array, pki, color='orange', alpha=0.7, label='Predicted Waveform')

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("PKI/ Actuals")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Saving plot
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {save_path}")

    plt.show()


def plot_preds_vs_act_freq(true_labels, predictions, freq_array, pki=None, title=None, save_path=None, close_figure: bool =True):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if title is not None:
        ax.set_title(title + ": Predictions vs. Actual", fontsize=16)
    else:
        ax.set_title("Predictions vs. Actual", fontsize=16)

    if pki is not None:
        ax.scatter(freq_array, pki, color='red', alpha=0.5, label='PKI')
    ax.scatter(freq_array, true_labels, color='blue', alpha=0.7, label='Actual Waveform')
    ax.scatter(freq_array, predictions, color='orange', alpha=0.7, label='Predicted Waveform')
    
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Predictions/ Actuals")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Saving plot
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {save_path}")

    if not close_figure:
        plt.show()
    if close_figure:
        plt.close(fig)


def plot_ber_vs_length(length_values, predictions, title=None):
    # Plots BER predictions as a function of length.
    #
    # Args:
    # - length_values: List of 1D arrays, one per curve (x-axis)
    # - predictions: List of 1D arrays, one per curve (BER values)
    # - title: Optional plot title
    # Returns:
    # *none*

    if not isinstance(length_values, list) or not isinstance(predictions, list):
        raise ValueError("length_values and predictions must both be lists of 1D arrays")
    if len(length_values) != len(predictions):
        raise ValueError("length_values and predictions must have the same number of curves")
    if len(length_values) == 0:
        raise ValueError("No curves to plot")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if title is not None:
        ax.set_title(title + ": BER vs. Length", fontsize=16)
    else:
        ax.set_title("BER vs. Length", fontsize=16)

    # Plot each curve with a distinct color
    color_map = plt.get_cmap('tab10')
    for curve_idx, (x_curve, y_curve) in enumerate(zip(length_values, predictions)):
        x_curve = np.asarray(x_curve).reshape(-1)
        y_curve = np.asarray(y_curve).reshape(-1)

        if x_curve.shape[0] != y_curve.shape[0]:
            raise ValueError(f"Curve {curve_idx} has mismatched lengths for x and y")

        ax.plot(
            x_curve,
            y_curve,
            marker='o',
            alpha=0.85,
            color=color_map(curve_idx % 10),
            label=f"Curve {curve_idx + 1}"
        )
    
    ax.set_xlabel("Length")
    ax.set_ylabel("BER")
    # ax.set_yscale('log')  # Use log scale for BER
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    
def plot_training_curves(train_losses, val_losses, train_maes, val_maes, title=None, save_path = None, close_figure: bool =True):
    # Plots training curves
    #
    # Args:
    # - train_losses: List of training losses per epoch
    # - val_losses: List of validation losses per epoch
    # - train_maes: List of training MAEs per epoch
    # - val_maes: List of validation MAEs per epoch
    # - title: Title of configuration
    # - save_path: Path where plot is saved
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

    # Saving plot
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {save_path}")

    if not close_figure:
        plt.show()
    if close_figure:
        plt.close(fig)


def plot_predicted_vs_actual(true_labels, predictions, title=None, save_path = None, close_figure: bool =True):
    # Plots predicted values against the corresponding actual values
    #
    # Args:
    # - true_labels: True target values
    # - predictions: Model predictions
    # - title: Optional plot title
    # - save_path: Path where plot is saved 
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

    # Saving plot
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {save_path}")

    if not close_figure:
        plt.show()
    if close_figure:
        plt.close(fig)


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

def plot_confusion_matrix(true_labels, predictions, title=None, class_names=None):
    # Plots a confusion matrix for classification models
    #
    # Args:
    # - true_labels: True class labels
    # - predictions: Predicted class labels
    # - title: Optional plot title
    # - class_names: Optional list of class names (e.g., ["Class A", "Class B", "Class C"])
    # Returns:
    # *none*

    true_labels = np.asarray(true_labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    if true_labels.shape != predictions.shape:
        raise ValueError("true_labels and predictions must have the same shape.")

    # If class names are provided, force a matrix for all class indices so the
    # plot shape stays consistent even when a class is absent in the current split.
    if class_names is not None:
        labels = np.arange(len(class_names))
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        num_classes = len(class_names)
    else:
        labels = np.unique(np.concatenate([true_labels, predictions]))
        cm = confusion_matrix(true_labels, predictions, labels=labels)
        num_classes = len(labels)
        class_names = [str(label) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 6))

    if title is not None:
        ax.set_title(title + ": Confusion Matrix", fontsize=16)
    else:
        ax.set_title("Confusion Matrix", fontsize=16)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()