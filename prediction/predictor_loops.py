import torch
import numpy as np

def train_pred_loop(model, data: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device):
    """
    # train_pred_loop()
    ## Loop used for training the model for one epoch
    
    ## Args:
    - model: The neural network model to be trained
    - data: Training data
    - optimizer: Optimization algorithm 
    - criterion: Loss function 
    - device: Device to run the training on (CPU or GPU)
    - complex_output: Indicates if output is complex
    ## Returns:
    - avg_loss: Average loss over the epoch
    - avg_mae: Average mean absolute error over the epoch
    - avg_mae_per_output: Average mae of each output 
    - avg_mae_real: Average mae of Re(Out)
    - avg_mae_imag: Average mae of Im(Out)
    """

    # Model in training mode
    model.train() 

    # Initialize metric sums
    total_loss = 0.0
    total_mae = 0.0 
    total_mae_real = 0.0
    total_mae_imag = 0.0
    total_samples = 0

    first_batch = True
    num_of_outputs = None 
    total_mae_per_output = None
    complex_outputs = None

    # Iterate through training batches
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


def test_pred_loop(model, data: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device):
    """
    # test_pred_loop()
    ## Loop used for evaluating the model
    
    ## Args:
    - model: The neural network model to be evaluated
    - data: Validation or Test data
    - criterion: Loss function
    - device: Device to run the evaluation on (CPU or GPU)
    ## Returns:
    - avg_loss: Average loss over the evaluation
    - avg_mae: Average mean absolute error over the evaluation
    - all_preds: All predictions made by the model
    - all_labels: All true labels corresponding to the predictions
    - avg_mae_per_output: Average mae for each output
    - avg_mape_per_output: Average mape for each output
    - avg_mae_real: Average mae for Re(Out)
    - avg_mae_imag: Average mae for Im(Out)
    - avg_mape_real: Average mape for Re(Out)
    - avg_mape_imag: Average mape for Im(Out)
    """

    # Model in evaluation mode
    model.eval() 

    # Initialize metric sums
    total_loss = 0.0
    total_mae = 0.0 
    total_mae_real = 0.0
    total_mae_imag = 0.0
    total_mape = 0.0
    total_mape_real = 0.0
    total_mape_imag = 0.0
    total_samples = 0
    num_of_outputs = None

    first_batch = True
    num_of_outputs = None
    complex_outputs = None
    total_mae_per_output = None
    total_mape_per_output = None

    all_preds = []
    all_labels = []

    eps = 1e-8 # to account for labels being equal to 0

    with torch.no_grad(): # disable gradient calculation
        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # loss function
            total_loss += loss.item() * inputs.size(0) 

            # Unsqueeze to (batch_size, 1)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

            if first_batch:
                num_of_outputs = outputs.size(1)
                complex_outputs = outputs.is_complex() 

                if complex_outputs:
                    total_mae_per_output = torch.zeros([num_of_outputs, 2], device=device)
                    total_mape_per_output = torch.zeros([num_of_outputs, 2], device=device)
                else:
                    total_mae_per_output = torch.zeros(num_of_outputs, device=device)
                    total_mape_per_output = torch.zeros(num_of_outputs, device=device)

                first_batch = False 

            total_samples += inputs.size(0)

            all_preds.append(outputs.cpu().numpy()) # store all predictions
            all_labels.append(labels.cpu().numpy()) # store all true labels

            if complex_outputs:
                # MAE
                real_error = torch.abs(outputs.real - labels.real)
                imag_error = torch.abs(outputs.imag - labels.imag)

                total_mae += torch.sum(real_error + imag_error).item()
                total_mae_real += torch.sum(real_error).item()
                total_mae_imag += torch.sum(imag_error).item()

                # MAPE
                mape_real = real_error / (torch.abs(labels.real) + eps)
                mape_imag = imag_error / (torch.abs(labels.imag) + eps)

                total_mape += torch.sum(mape_real + mape_imag).item()
                total_mape_real += torch.sum(mape_real).item()
                total_mape_imag += torch.sum(mape_imag).item()

                total_mae_per_output[:, 0] += torch.sum(real_error, dim=0)
                total_mae_per_output[:, 1] += torch.sum(imag_error, dim=0)
                total_mape_per_output[:, 0] += torch.sum(mape_real, dim=0)
                total_mape_per_output[:, 1] += torch.sum(mape_imag, dim=0)
            else:
                error = torch.abs(outputs - labels)
                
                total_mae += torch.sum(error).item()
                total_mae_real += torch.sum(error).item()
                total_mape += torch.sum(error / (torch.abs(labels) + eps)).item()

                total_mae_per_output += torch.sum(error, dim=0)

    avg_loss = total_loss / total_samples

    total_elements = total_samples * num_of_outputs

    avg_mae = total_mae / total_elements
    avg_mae_real = total_mae_real / total_elements
    avg_mae_imag = total_mae_imag / total_elements
    avg_mae_per_output = (total_mae_per_output / total_samples).cpu().tolist()
    
    avg_mape = total_mape / total_elements
    avg_mape_real = total_mape_real / total_elements
    avg_mape_imag = total_mape_imag / total_elements
    avg_mape_per_output = (total_mape_per_output / total_samples).cpu().tolist()

    final_preds = np.concatenate(all_preds, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    return avg_loss, avg_mae, avg_mape, final_preds, final_labels, avg_mae_per_output, avg_mape_per_output, avg_mae_real, \
        avg_mae_imag, avg_mape_real, avg_mape_imag
