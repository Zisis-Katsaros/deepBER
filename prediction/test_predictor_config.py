from prediction.predictor_loops import train_pred_loop, test_pred_loop
from visualization import plot_error_distribution, plot_error_vs_feature, plot_predicted_vs_actual, plot_training_curves, plot_ber_vs_length
import prediction.predictor as predictor
import torch
import numpy as np

def test_predictor_configuration(title, device, model, dataloader, learning_rate, batch_size, criterion, optimizer, scheduler=None,
                       epochs=30, early_stopping=False, patience=5, training_curves=False,
                       predicted_vs_actual=False, error_distribution=False, error_vs_feature=None,
                       feature_columns=None, output_names = None):
    # Train model with given configuration 
    #
    # Args:
    # - title: Title of the test configuration (for visualization and logging)
    # - device: Device to run the model on (CPU or GPU)
    # - model: The neural network model to be trained and evaluated
    # - dataloader: [train_data, val_data, test_data]
    # - learning_rate: Learning rate for the optimizer
    # - batch_size: Batch size for training and evaluation
    # - criterion: Loss function
    # - optimizer: Optimization algorithm
    # - epochs: Maximum number of training epochs
    # - early_stopping: If true training stops if there is no improvement
    # - patience: Number of epochs to wait for improvement before stopping (if early_stopping is true)
    # - training_curves: If true training curves will be plotted at the end of training
    # - predicted_vs_actual: If true a plot of predicted vs. actual values will be plotted at the end of training
    # - error_distribution: If true a histogram of prediction errors will be plotted at the end of training
    # - error_vs_feature: List of feature names for which to plot error vs. feature
    # - feature_columns: List of feature names matching the model input columns
    # Returns:
    # *none*

    # Print test details
    print(f'Using device: {device}\n\n')

    print(f"{title}")
    print(f" Info:")
    print(f" - Model:")
    print(model)
    print(f" - Data Split:")
    print(f"   - Train: {len(dataloader[0].dataset)} samples")
    print(f"   - Validation: {len(dataloader[1].dataset)} samples")
    print(f"   - Test: {len(dataloader[2].dataset)} samples")
    print(f" - Criterion: {criterion}")
    print(f" - Learning Rate: {learning_rate}")
    print(f" - Batch Size: {batch_size}")
    print(f" - Epochs: {epochs}")
    print(f" - Early Stopping: {early_stopping}")
    if early_stopping:
        print(f" - Patience: {patience}")

    # Initialize data splits
    train_data = dataloader[0]
    val_data = dataloader[1]
    test_data = dataloader[2]
    
    # Initialize metric lists
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    # Initialize early stopping parameters
    best_val_loss = float('inf')
    best_model_epoch = None
    counter = 0

    print(f"==================== Starting Training ====================")
    for epoch in range(epochs):
        train_loss, train_mae = train_pred_loop(model, train_data, optimizer, criterion, device)
        val_loss, val_mae, _, _, _ = test_pred_loop(model, val_data, criterion, device)

        # Step scheduler
        if scheduler is not None:
            try:    
                scheduler.step(val_loss)
            except TypeError:
                scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        # Early Stopping Check
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_epoch = epoch + 1
                counter = 0
                # save model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Best model at epoch {best_model_epoch}")
                    model.load_state_dict(torch.load('best_model.pth'))
                    break

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}")
            print(f" - Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}")
            print(f" - Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}\n")

    print(f"==================== Training complete ====================")

    test_loss, test_mae, test_mape, test_preds, test_targets = test_pred_loop(model, test_data, criterion, device)
    print(f">>> Test MAE: {test_mae:.6f}")
    print(f">>> Test MAPE: {test_mape*100:.4f}%")

    num_outputs = test_preds.shape[1] if test_preds.ndim > 1 else 1
    if output_names is None:
        output_names = [f"Output Target {i+1}" for i in range(num_outputs)]

    if num_outputs > 1:
        print("\n>>> Performance Breakdown per Output Target:")
        for i in range(num_outputs):
            name = output_names[i]
            col_mae = np.mean(np.abs(test_preds[:, i] - test_targets[:, i]))
            
            # Prevent division by zero if targets contain 0
            with np.errstate(divide='ignore', invalid='ignore'):
                col_mape = np.mean(np.abs((test_preds[:, i] - test_targets[:, i]) / test_targets[:, i])) * 100
            
            print(f" - {name} -> MAE: {col_mae:.6f} | MAPE: {col_mape:.4f}%")

    # Visualization
    if training_curves:
        plot_training_curves(train_losses, val_losses, train_maes, val_maes, title=title)
    
    if predicted_vs_actual:
        plot_predicted_vs_actual(test_targets, test_preds, title=title)
    
    if error_distribution:
        plot_error_distribution(test_targets, test_preds, title=title)
    
    test_features = test_data.dataset.tensors[0].cpu().numpy()
    if error_vs_feature:
        if feature_columns is None:
            raise ValueError("feature_columns must be provided when error_vs_feature is used.")

        for feature in error_vs_feature:
            if isinstance(feature, str):
                if feature not in feature_columns:
                    raise ValueError(f"Feature '{feature}' was not found in feature_columns.")
                feature_index = feature_columns.index(feature)
                feature_name = feature
            else:
                feature_index = int(feature)
                if feature_index < 0 or feature_index >= len(feature_columns):
                    raise ValueError(f"Feature index {feature_index} is out of range.")
                feature_name = feature_columns[feature_index]

            plot_error_vs_feature(
                test_features[:, feature_index],
                test_targets,
                test_preds,
                title=title,
                feature_name=feature_name,
            )


def ber_vs_length_test(model, feature_arrays, length_interval, number_of_points, feature_columns,
                       device='cpu', logBER=True, visualization=False, title=None,
                       feature_mean=None, feature_std=None):  
    # Test how BER varies with length by fixing all other features.
    #
    # Args:
    # - model: Trained model to use for prediction
    # - feature_arrays: List of 1D arrays, one per curve. Each array contains
    #   all fixed feature values except 'length'.
    # - length_interval: [min_length, max_length] tuple or list
    # - number_of_points: Number of length values to test
    # - feature_columns: List of all feature column names (including 'length')
    # - device: Device to run inference on ('cpu' or 'cuda')
    # - logBER: If True, model predicts log10(BER); if False, raw BER
    # - visualization: If True, plot the results
    # - title: Title for the visualization
    # - feature_mean: 1D array of mean values per feature (for standardization)
    # - feature_std: 1D array of std values per feature (for standardization)
    # Returns:
    # - length_values: List of 1D arrays (x values), one per curve
    # - ber_predictions: List of 1D arrays (predicted BER), one per curve

    # Expect `feature_arrays` to be a list of 1D arrays (one per curve)
    if not isinstance(feature_arrays, list):
        raise ValueError("feature_arrays must be a list of 1D arrays (one per curve)")

    # Convert entries to 1D numpy arrays
    feature_arrays = [np.asarray(arr, dtype=np.float32).reshape(-1) for arr in feature_arrays]

    # Find the index of 'length' in feature_columns
    if 'length' not in feature_columns:
        raise ValueError("'length' not found in feature_columns")
    length_index = feature_columns.index('length')

    # Validate inputs
    if len(feature_arrays) == 0:
        raise ValueError("feature_arrays is empty")
    expected_feature_count = len(feature_columns) - 1
    for curve_idx, curve_features in enumerate(feature_arrays):
        if curve_features.shape[0] != expected_feature_count:
            raise ValueError(
                f"Curve {curve_idx} must contain exactly {expected_feature_count} values "
                f"(all features except 'length'); got {curve_features.shape[0]}"
            )
    
    # Create length values to test
    min_length, max_length = length_interval
    length_values = np.linspace(min_length, max_length, number_of_points)
    
    n_features_total = len(feature_columns)
    
    # Validate and normalize scaling parameters
    if feature_mean is not None:
        feature_mean = np.asarray(feature_mean, dtype=np.float32)
        if feature_mean.shape[0] != n_features_total:
            raise ValueError(f"feature_mean must have {n_features_total} values")
    if feature_std is not None:
        feature_std = np.asarray(feature_std, dtype=np.float32)
        if feature_std.shape[0] != n_features_total:
            raise ValueError(f"feature_std must have {n_features_total} values")
        feature_std = np.where(feature_std == 0.0, 1.0, feature_std)
    
    # Prepare model for inference
    model.eval()
    
    # Store one length/prediction array pair per curve
    all_length_values = []
    all_predictions = []
    
    with torch.no_grad():
        for curve_features in feature_arrays:
            batch_array = np.zeros((number_of_points, n_features_total), dtype=np.float32)

            # Fill all non-length features for this curve
            feature_idx = 0
            for col_idx in range(n_features_total):
                if col_idx == length_index:
                    continue
                batch_array[:, col_idx] = curve_features[feature_idx]
                feature_idx += 1

            # Sweep length
            batch_array[:, length_index] = length_values

            # Apply standardization if parameters provided
            if feature_mean is not None and feature_std is not None:
                batch_array = (batch_array - feature_mean) / feature_std

            batch_tensor = torch.from_numpy(batch_array).to(device)
            predictions = model(batch_tensor).cpu().numpy().reshape(-1)

            # Convert from log BER if needed
            if logBER:
                predictions = 10 ** predictions

            all_length_values.append(length_values.copy())
            all_predictions.append(predictions)
    
    # Visualization
    if visualization:
        plot_ber_vs_length(all_length_values, all_predictions, title=title)
    
    return all_length_values, all_predictions

