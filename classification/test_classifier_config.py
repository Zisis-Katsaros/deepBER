import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from classification.ber_to_class import ber_to_class
from classification.classifier_loops import train_classifier_loop, test_classifier_loop, train_xgb_loop, test_xgb_loop
from visualization import plot_confusion_matrix



def test_classifier_configuration(
    title,
    model,
    dataloader,
    lower_thres=10**(-5.5),
    upper_thres=10**(-2.5),
    weight=False,
    device=None,
    learning_rate=1e-3,
    batch_size=None,
    criterion=None,
    optimizer=None,
    epochs=30,
    early_stopping=False,
    patience=5,
    confusion_matrix=False,
    class_names=None, 
):
    train_data, val_data, test_data = dataloader

    if isinstance(model, torch.nn.Module):
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        if batch_size is not None:
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
        train_accs = []
        val_accs = []
        train_f1s = []
        val_f1s = []

        # Initialize early stopping parameters
        best_val_f1 = float('-inf')
        best_model_epoch = None
        counter = 0

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        print(f"==================== Starting Training ====================")
        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = train_classifier_loop(model, train_data, optimizer, criterion, device)
            val_loss, val_acc, val_f1, _, _ = test_classifier_loop(model, val_data, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

            # Early Stopping Check (based on F1 score)
            if early_stopping:
                if val_f1 > best_val_f1 + 1e-8:
                    best_val_f1 = val_f1
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
                print(f" - Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc*100:.2f}%, Train F1: {train_f1:.6f}")
                print(f" - Val Loss: {val_loss:.6f}, Val Accuracy: {val_acc*100:.2f}%, Val F1: {val_f1:.6f}\n")

        print(f"==================== Training complete ====================")

        test_loss, test_acc, test_f1, test_preds, test_labels = test_classifier_loop(model, test_data, criterion, device)  

        trained_model = model

    else:
        trained_model = train_xgb_loop(
            model,
            train_data,
            val_data=val_data,
            label_transform=ber_to_class,
            lower_thres=lower_thres,
            upper_thres=upper_thres,
            weight=weight,
        )

        test_acc, _, test_preds, test_labels = test_xgb_loop(
            trained_model,
            test_data,
            label_transform=ber_to_class,
            lower_thres=lower_thres,
            upper_thres=upper_thres,
        )
        test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)

    print(f">>> Test Accuracy: {test_acc*100:.2f}%")
    if isinstance(model, torch.nn.Module):
        print(f">>> Test F1 Score: {test_f1:.6f}")

    if confusion_matrix:
        if class_names is None:
            class_names = ["Feasible", "Uncertain", "Infeasible"]
        plot_confusion_matrix(test_labels, test_preds, title=title, class_names=class_names)

    return trained_model, test_acc

