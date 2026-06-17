import torch
import numpy as np

def train_pred_loop(model, data, optimizer, criterion, device):
    # Loop used for training the model for one epoch
    #
    # Args:
    # - model: The neural network model to be trained
    # - data: Training data
    # - optimizer: Optimization algorithm 
    # - criterion: Loss function 
    # - device: Device to run the training on (CPU or GPU)
    # - complex_output: Indicates if output is complex
    # Returns:
    # - avg_loss: Average loss over the epoch
    # - avg_mae: Average mean absolute error over the epoch

    model.train() # model in training mode

    total_loss = 0.0
    total_mae = 0.0 
    total_mae_real = 0.0
    total_mae_imag = 0.0
    total_samples = 0

    first_batch = True
    num_of_outputs = None 
    total_mae_per_output = None
    complex_outputs = None

    # iterate through training batches
    for inputs, labels in data:
        inputs = inputs.to(device) 
        labels = labels.to(device)

        outputs = model(inputs) # forward pass
        loss = criterion(outputs, labels) # loss function

        optimizer.zero_grad() # zero the previous gradients
        loss.backward() # backpropagation
        optimizer.step() # update weights

        # Unsqueeze to (batch_size, 1)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # Grab info from the first batch
        if first_batch:
            num_of_outputs = outputs.size(1)
            complex_outputs = outputs.is_complex()
            
            if complex_outputs:
                total_mae_per_output = torch.zeros([num_of_outputs,2], device=device)
            else:
                total_mae_per_output = torch.zeros(num_of_outputs, device=device)
            
            first_batch = False

        total_loss += loss.item() * inputs.size(0) # accumulate loss
        total_samples += inputs.size(0) # accumulate number of samples

        if complex_outputs:
            real_error = torch.abs(outputs.real - labels.real)
            imag_error = torch.abs(outputs.imag - labels.imag)

            total_mae += torch.sum(real_error + imag_error).item()
            total_mae_real += torch.sum(real_error).item()
            total_mae_imag += torch.sum(imag_error).item()

            # Fix 6: Keep tracking variables as tensors during loop (removed .item())
            total_mae_per_output[:, 0] += torch.sum(real_error, dim=0)
            total_mae_per_output[:, 1] += torch.sum(imag_error, dim=0)
        else:
            error = torch.abs(outputs - labels)

            total_mae += torch.sum(error).item()
            total_mae_real += torch.sum(error).item()

            total_mae_per_output += torch.sum(error, dim=0)

        
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / (total_samples*num_of_outputs)
    avg_mae_real = total_mae_real / (total_samples*num_of_outputs)
    avg_mae_imag = total_mae_imag / (total_samples*num_of_outputs)
    avg_mae_per_output = (total_mae_per_output / total_samples).cpu().tolist()

    return avg_loss, avg_mae, avg_mae_per_output, avg_mae_real, avg_mae_imag # return loss and MAE

# Test Loop:
def test_pred_loop(model, data, criterion, device):
    # Loop used for evaluating the model
    # 
    # Args:
    # - model: The neural network model to be evaluated
    # - data: Validation or Test data
    # - criterion: Loss function
    # - device: Device to run the evaluation on (CPU or GPU)
    # Returns:
    # - avg_loss: Average loss over the evaluation
    # - avg_mae: Average mean absolute error over the evaluation
    # - all_preds: All predictions made by the model
    # - all_labels: All true labels corresponding to the predictions

    model.eval() # model in evaluation mode

    total_loss = 0.0
    total_mae = 0.0 
    total_mape = 0.0
    total_samples = 0
    num_of_outputs = None

    all_preds = []
    all_labels = []

    with torch.no_grad(): # disable gradient calculation
        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # loss function
            total_loss += loss.item() * inputs.size(0) 

            if num_outputs is None:
                num_outputs = outputs.size(1) if outputs.dim() > 1 else 1

            total_mae += torch.sum(torch.abs(outputs - labels)).item()
            total_mape += torch.sum(torch.abs((outputs - labels) / labels)).item()
            total_samples += inputs.size(0)

            all_preds.extend(outputs.cpu().numpy()) # store all predictions
            all_labels.extend(labels.cpu().numpy()) # store all true labels

    avg_loss = total_loss / total_samples

    total_elements = total_samples * num_of_outputs
    avg_mae = total_mae / total_elements
    avg_mape = total_mape / total_elements

    final_preds = np.concatenate(all_preds, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    return loss, avg_mae, avg_mape, final_preds, final_labels 
