from prediction.predictor_loops import train_pred_loop, test_pred_loop
from visualization import plot_error_distribution, plot_error_vs_feature, plot_predicted_vs_actual, plot_training_curves, plot_ber_vs_length, plot_preds_vs_act_freq
from load_set import get_grouping
import torch
import numpy as np
import os
from sklearn.metrics import mean_absolute_error

def mae(labels, preds):
    is_complex = np.iscomplexobj(labels)
    if not is_complex:
        return mean_absolute_error(labels, preds)
    else:
        real_error = np.abs(labels.real - preds.real)
        imag_error = np.abs(labels.imag - preds.imag)
        return np.mean(real_error + imag_error)


def smape(labels, preds, epsilon=1e-8):
    numerator = np.abs(preds - labels)
    denominator = (np.abs(labels) + np.abs(preds)) / 2.0
    return np.mean(numerator / (denominator + epsilon))


def test_predictor_configuration(title: str, device: torch.device, model, dataloader: list[torch.utils.data.DataLoader], learning_rate: float, 
                                batch_size: int, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler=None, epochs: int =30, 
                                early_stopping: bool =False, patience: int =5, y_scale_params: tuple =None, training_curves: bool =False,
                                predicted_vs_actual: bool =False, error_distribution: bool =False, error_vs_feature: bool =None,
                                feature_columns=None, output_names = None, test_out_dir: str ='.'):
    """ 
    # test_predictor_configuration()
    ## Train model with given configuration and visualize/ save results
    
    ## Args:
    - title: Title of the test configuration (for visualization and logging)
    - device: Device to run the model on (CPU or GPU)
    - model: The neural network model to be trained and evaluated
    - dataloader: [train_data, val_data, test_data]
    - learning_rate: Learning rate for the optimizer
    - batch_size: Batch size for training and evaluation
    - criterion: Loss function
    - optimizer: Optimization algorithm
    - epochs: Maximum number of training epochs
    - early_stopping: If true training stops if there is no improvement
    - patience: Number of epochs to wait for improvement before stopping (if early_stopping is true)
    - training_curves: If true training curves will be plotted at the end of training
    - predicted_vs_actual: If true a plot of predicted vs. actual values will be plotted at the end of training
    - error_distribution: If true a histogram of prediction errors will be plotted at the end of training
    - error_vs_feature: List of feature names for which to plot error vs. feature
    - feature_columns: List of feature names matching the model input columns
    - output_names: Name of each output incase of multiple outputs
    - test_out_dir: Directory where output files are saved
    ## Returns:
    *none*
    """

    # Print test details 
    print(f" Info:")
    print(f'Using device: {device}')
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

    # Create output directories
    os.makedirs(test_out_dir, exist_ok=True)
    model_save_path = os.path.join(test_out_dir, "best_model.pth")
    training_curves_save_path = os.path.join(test_out_dir, "training_curves.png")
    results_save_path = os.path.join(test_out_dir, "test_results.npz")

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
        train_loss, train_mae, *_ = train_pred_loop(model, train_data, optimizer, criterion, device)
        val_loss, val_mae, *_ = test_pred_loop(model, val_data, criterion, device)

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
                torch.save(model.state_dict(), model_save_path)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Best model at epoch {best_model_epoch}")
                    model.load_state_dict(torch.load(model_save_path))
                    break

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}")
            print(f" - Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}")
            print(f" - Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}\n")

    print(f"==================== Training complete ====================")

    _, _, _, test_preds, test_targets, *_, = test_pred_loop(model, test_data, criterion, device)

    if y_scale_params is not None:
        if torch.is_tensor(test_preds):
            test_preds = test_preds.cpu().numpy()
            test_targets = test_targets.cpu().numpy()
            
        test_preds = test_preds * y_scale_params[1] + y_scale_params[0]
        test_targets = test_targets * y_scale_params[1] + y_scale_params[0]
    np.savez_compressed(results_save_path, preds=test_preds, targets=test_targets)

    is_complex = np.iscomplexobj(test_preds)
    
    test_mae = mae(test_targets, test_preds)
    test_smape = smape(test_targets, test_preds)
    print(f">>> Test MAE: {test_mae:.6f}")
    print(f">>> Test sMAPE: {test_smape * 100:.4f}%")
    
    num_outputs = test_preds.shape[1] if test_preds.ndim > 1 else 1
    if output_names is None:
        output_names = [f"Out{i+1}" for i in range(num_outputs)]

    
    if is_complex:
        mae_real = mean_absolute_error(np.real(test_targets), np.real(test_preds))
        smape_real = smape(np.real(test_targets), np.real(test_preds))
        mae_imag = mean_absolute_error(np.imag(test_targets), np.imag(test_preds))
        smape_imag = smape(np.imag(test_targets), np.imag(test_preds))
        
        print("\n>>> Overall Complex Performance:")
        print(f" - Real -> MAE: {mae_real:.6f} | sMAPE: {smape_real * 100:.4f}%")
        print(f" - Imag -> MAE: {mae_imag:.6f} | sMAPE: {smape_imag * 100:.4f}%")
        
    elif num_outputs > 1:
        print("\n>>> Performance Breakdown per Output Target:")
        for i in range(num_outputs):
            name = output_names[i]
            y_true_col = test_targets[:, i]
            y_pred_col = test_preds[:, i]
            
            out_mae = mean_absolute_error(y_true_col, y_pred_col)
            out_smape = smape(y_true_col, y_pred_col)
            
            print(f" - {name} -> MAE: {out_mae:.6f} | sMAPE: {out_smape * 100:.4f}%")

    # Visualization
    if training_curves:
        plot_training_curves(train_losses, val_losses, train_maes, val_maes, title=title, save_path=training_curves_save_path)
    
    if predicted_vs_actual:
        if num_outputs > 1:
            for out_idx in range(num_outputs):
                title_out = f"{title} - {output_names[out_idx]}"
                pred_vs_act_save_path = os.path.join(test_out_dir, f"pred_vs_actual_out{out_idx}.png")
                plot_predicted_vs_actual(test_targets[:, out_idx], test_preds[:, out_idx], title=title_out, save_path=pred_vs_act_save_path)
        elif test_targets[0].dtype == np.complex64:
            title_real = f"{title} - Real Part"
            title_imag = f"{title} - Imaginary Part"
            pred_vs_act_save_path_real = os.path.join(test_out_dir, "pred_vs_actual_real.png")
            pred_vs_act_save_path_imag = os.path.join(test_out_dir, "pred_vs_actual_imag.png")
            
            plot_predicted_vs_actual(test_targets.real, test_preds.real, title=title_real, save_path=pred_vs_act_save_path_real)
            plot_predicted_vs_actual(test_targets.imag, test_preds.imag, title=title_imag, save_path=pred_vs_act_save_path_imag)
        else:
            pred_vs_act_save_path = os.path.join(test_out_dir, "pred_vs_actual.png")
            plot_predicted_vs_actual(test_targets, test_preds, title=title, save_path=pred_vs_act_save_path)
    
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

def single_geometry_test(title: str, device: torch.device, model, test_data: torch.utils.data.DataLoader, x_scale_params, y_scale_params, max_geoms: int =None, 
                         pki: bool =False, n_non_unique_feats: int =7, save_dir=None):
    model.eval()
    
    # Create output directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Extract all inputs and true labels from the DataLoader
    x_list, y_list = [], []
    for inputs, labels in test_data:
        x_list.append(inputs.numpy())
        y_list.append(labels.numpy())
        
    x_array = np.concatenate(x_list, axis=0)
    y_array = np.concatenate(y_list, axis=0)
    
    # Perform forward pass to get all predictions
    preds_list = []
    with torch.no_grad():
        for inputs, _ in test_data:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_list.append(outputs.cpu().numpy())
    preds_array = np.concatenate(preds_list, axis=0)
    
    # If arrays are 1D (batch_size,), unsqueeze them to (batch_size, 1)
    if y_array.ndim == 1: 
        y_array = np.expand_dims(y_array, 1)
    if preds_array.ndim == 1: 
        preds_array = np.expand_dims(preds_array, 1)

    # Unscale the data to real-world units
    x_unscaled = x_array * x_scale_params[1] + x_scale_params[0]
    y_unscaled = y_array * y_scale_params[1] + y_scale_params[0]
    preds_unscaled = preds_array * y_scale_params[1] + y_scale_params[0]
    
    # Group data by unique geometric parameters
    grouping_indices, _ = get_grouping(x_unscaled, n_non_unique_feats=n_non_unique_feats)
    
    freq_idx = n_non_unique_feats 
    num_outputs = y_array.shape[1]

    # Loop through each geometry group and plot
    for i, indices in enumerate(grouping_indices):
        if max_geoms is not None and i >= max_geoms:
            break
            
        group_x = x_unscaled[indices]
        group_y = y_unscaled[indices]
        group_preds = preds_unscaled[indices]
        
        # Extract Frequency 
        freq_array = group_x[:, freq_idx]
        sort_idx = np.argsort(freq_array)
        
        freq_array = freq_array[sort_idx]
        group_x = group_x[sort_idx]
        group_y = group_y[sort_idx]
        group_preds = group_preds[sort_idx]
        
        # Iterate over output elements 
        for out_idx in range(num_outputs):
            y_true_col = group_y[:, out_idx]
            y_pred_col = group_preds[:, out_idx]
            
            # Plotting
            # If the outputs are complex 
            if np.iscomplexobj(y_true_col):
                pki_real, pki_imag = None, None
                if pki:
                    pki_real = group_x[:, -2 * num_outputs + 2 * out_idx]
                    pki_imag = group_x[:, -2 * num_outputs + 2 * out_idx + 1]

                # Plot Real part
                if save_dir is not None:
                    plot_save_path_re = os.path.join(save_dir, f"pred_vs_act_freq_Re_{i+1}")
                    plot_save_path_im = os.path.join(save_dir, f"pred_vs_act_freq_Im_{i+1}")
                else:
                    plot_save_path_re, plot_save_path_im = None, None
                plot_preds_vs_act_freq(
                    true_labels=np.real(y_true_col),
                    predictions=np.real(y_pred_col),
                    freq_array=freq_array,
                    pki=pki_real,
                    title=f"{title} | Geom {i+1} [Real]" if num_outputs == 1 else f"{title} | Geom {i+1} Out{out_idx+1} [Real]",
                    save_path=plot_save_path_re
                )
                # Plot Imaginary part
                plot_preds_vs_act_freq(
                    true_labels=np.imag(y_true_col),
                    predictions=np.imag(y_pred_col),
                    freq_array=freq_array,
                    pki=pki_imag,
                    title=f"{title} | Geom {i+1} [Imag]" if num_outputs == 1 else f"{title} | Geom {i+1} Out{out_idx+1} [Imag]",
                    save_path=plot_save_path_im
                )    
            # If outputs are single or dual
            else:
                pki_feat = None
                if pki:
                    pki_feat = group_x[:, -num_outputs + out_idx] 

                if save_dir is not None:
                    plot_save_path = os.path.join(save_dir, f"pred_vs_act_freq_Out{out_idx+1}_{i+1}")
                else:
                    plot_save_path = None
                plot_preds_vs_act_freq(
                    true_labels=y_true_col,
                    predictions=y_pred_col,
                    freq_array=freq_array,
                    pki=pki_feat,
                    title=f"{title} | Geom {i+1}" if num_outputs == 1 else f"{title} | Geom {i+1} - Out{out_idx+1}",
                    save_path=plot_save_path
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

