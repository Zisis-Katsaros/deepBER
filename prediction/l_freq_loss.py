import torch
import torch.nn as nn

class l_freq_loss(nn.Module):
    def __init__(self, eps=1e-12):
        super(l_freq_loss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2

        # Compute the mean accross frequency dimension
        mse_per_response = torch.mean(squared_error, dim=list(range(1, squared_error.ndim)))

        # Take the square root
        rmse_per_response = torch.sqrt(mse_per_response + self.eps)

        # Average over N samples in batch
        loss = torch.mean(rmse_per_response)
        
        return loss