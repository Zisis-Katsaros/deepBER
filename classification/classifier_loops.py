import numpy as np
import torch
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

def loader_to_numpy(data_loader):
    # Convert a torch DataLoader into NumPy arrays for non-torch models.
    tensors = data_loader.dataset.tensors
    features = tensors[0].cpu().numpy()
    labels = tensors[1].cpu().numpy().reshape(-1)
    return features, labels

def train_classifier_loop(model, data, optimizer, criterion, device):
    # Loop used for training the model for one epoch
    #
    # Args:
    # - model: The neural network model to be trained
    # - data: Training data
    # - optimizer: Optimization algorithm 
    # - criterion: Loss function 
    # - device: Device to run the training on (CPU or GPU)
    # Returns:
    # - avg_loss: Average loss over the epoch
    # - avg_acc: Average accuracy over the epoch
    # - avg_f1: Average F1 score over the epoch

    model.train() # model in training mode

    total_loss = 0.0
    total_acc = 0.0 
    total_samples = 0
    all_preds = []
    all_labels = []
    
    # iterate through training batches
    for inputs, labels in data:
        inputs = inputs.to(device) 
        labels = labels.to(device).view(-1).long()

        outputs = model(inputs) # forward pass
        loss = criterion(outputs, labels) # loss function

        optimizer.zero_grad() # zero the previous gradients
        loss.backward() # backpropagation
        optimizer.step() # update weights

        total_loss += loss.item() * inputs.size(0) # accumulate loss
        total_acc += torch.sum(torch.argmax(outputs, dim=1) == labels).item() # accumulate accuracy
        total_samples += inputs.size(0) # accumulate number of samples
        
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    avg_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, avg_acc, avg_f1 # return loss, accuracy and F1 score

def test_classifier_loop(model, data, criterion, device):
    # Loop used for evaluating the model
    # 
    # Args:
    # - model: The neural network model to be evaluated
    # - data: Validation or Test data
    # - criterion: Loss function
    # - device: Device to run the evaluation on (CPU or GPU)
    # Returns:
    # - avg_loss: Average loss over the evaluation
    # - avg_acc: Average accuracy over the evaluation
    # - avg_f1: Average F1 score over the evaluation
    # - all_preds: All predictions made by the model
    # - all_labels: All true labels corresponding to the predictions

    model.eval() # model in evaluation mode

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad(): # disable gradient calculation
        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1).long()

            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # loss function
            total_loss += loss.item() * inputs.size(0) 

            preds = torch.argmax(outputs, dim=1)
            maks = (preds== 0) & (outputs[:,0] <0.85)
            preds[maks] = 1

            total_acc += torch.sum(preds == labels).item()

            total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy()) # store all predictions
            all_labels.extend(labels.cpu().numpy()) # store all true labels

    loss = total_loss / total_samples
    acc = total_acc / total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    return loss, acc, f1, all_preds, all_labels


def train_xgb_loop(
    model,
    train_data,
    val_data=None,
    label_transform=None,
    lower_thres=10**(-5.5),
    upper_thres=10**(-2.5),
    weight=False,
):
    # Train an XGBoost classifier on train_data and optional validation data.
    #
    # Args:
    # - model: Configured XGBoost classifier instance
    # - train_data: Training DataLoader
    # - val_data: Optional validation DataLoader
    # - label_transform: Optional callable to convert labels
    # Returns:
    # - model: Trained model

    x_train, y_train = loader_to_numpy(train_data)

    if label_transform is not None:
        y_train = label_transform(y_train, lower_thres=lower_thres, upper_thres=upper_thres)

    fit_kwargs = {}

    if weight is True:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        fit_kwargs["sample_weight"] = sample_weight

    if val_data is not None:
        x_val, y_val = loader_to_numpy(val_data)
        if label_transform is not None:
            y_val = label_transform(y_val, lower_thres=lower_thres, upper_thres=upper_thres)
        fit_kwargs["eval_set"] = [(x_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(x_train, y_train, **fit_kwargs)
    return model


def test_xgb_loop(model, data, label_transform=None, lower_thres=10**(-5.5), upper_thres=10**(-2.5)):
    # Evaluate an XGBoost classifier.
    #
    # Args:
    # - model: Trained XGBoost classifier
    # - data: Validation or test DataLoader
    # - label_transform: Optional callable to convert labels
    # Returns:
    # - accuracy: Overall accuracy
    # - class_accuracy: Dict with per-class accuracy for classes 0,1,2
    # - preds: Predicted class labels
    # - labels: True class labels

    features, labels = loader_to_numpy(data)
    if label_transform is not None:
        labels = label_transform(labels, lower_thres=lower_thres, upper_thres=upper_thres)

    preds = model.predict(features)
    accuracy = float(np.mean(preds == labels))

    class_accuracy = {}
    for class_id in [0, 1, 2]:
        class_mask = labels == class_id
        if np.any(class_mask):
            class_accuracy[class_id] = float(np.mean(preds[class_mask] == labels[class_mask]))
        else:
            class_accuracy[class_id] = float("nan")

    return accuracy, class_accuracy, preds, labels