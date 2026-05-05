from loops import train_loop, test_loop
from visualization import plot_error_distribution, plot_error_vs_feature, plot_predicted_vs_actual, plot_training_curves
import model
import torch

def test_configuration(title, device, model, dataloader, learning_rate, batch_size, criterion, optimizer,
                       epochs=30, early_stopping=False, patience=5, training_curves=False,
                       predicted_vs_actual=False, error_distribution=False, error_vs_feature=None,
                       feature_columns=None):
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
        train_loss, train_mae = train_loop(model, train_data, optimizer, criterion, device)
        val_loss, val_mae, targets, preds = test_loop(model, val_data, criterion, device)

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

    _, test_mae, test_targets, test_preds = test_loop(model, test_data, criterion, device)
    print(f">>> Test MAE: {test_mae:.6f}")

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
        