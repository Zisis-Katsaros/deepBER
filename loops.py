import torch
import numpy as np

# Training Loop:
def train_loop(model, dataloader, optimizer, criterion, device):
    model.train() # model in training mode

    total_loss = 0.0
    total_mae = 0.0 # mean absolute error
    total_samples = 0
    
    # iterate through training batches
    for inputs, labels in dataloader:
        inputs = inputs.to(device) 
        labels = labels.to(device).view(-1,1)

        outputs = model(inputs) # forward pass
        loss = criterion(outputs, labels) # loss function

        optimizer.zero_grad() # zero the previous gradients
        loss.backward() # backpropagation
        optimizer.step() # update weights

        total_loss += loss.item() * inputs.size(0) # accumulate loss
        total_mae += torch.sum(torch.abs(outputs - labels)).item() # accumulate absolute error
        total_samples += inputs.size(0) # accumulate number of samples

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples

    return avg_loss, avg_mae # return loss and MAE

# Test Loop:
def test_loop(model, dataloader, criterion, device):
    model.eval() # model in evaluation mode

    total_loss = 0.0
    total_mae = 0.0 
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad(): # disable gradient calculation
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1,1)

            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # loss function
            total_loss += loss.item() * inputs.size(0) 

            total_mae += torch.sum(torch.abs(outputs - labels)).item()
            total_samples += inputs.size(0)

            all_preds.extend(outputs.cpu().squeeze().numpy()) # store all predictions
            all_labels.extend(labels.cpu().squeeze().numpy()) # store all true labels

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples

    # return loss, accuracy, all predictions and all true labels
    return avg_loss, avg_mae, np.array(all_preds), np.array(all_labels) 
