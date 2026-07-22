import torch
import torch.nn as nn

class l_freq_loss(nn.Module):
    def __init__(self, eps=1e-12, weight=None):
        super(l_freq_loss, self).__init__()
        self.eps = eps

        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None
            
    def forward(self, y_pred, y_true, weight=None):
        squared_error = (y_pred - y_true) ** 2

        # Compute the mean accross frequency dimension
        mse_per_response = torch.mean(squared_error, dim=2)

        # Take the square root
        rmse_per_response = torch.sqrt(mse_per_response + self.eps)

        # Apply weighting
        if weight is not None:
            rmse_per_response = rmse_per_response * weight

        # Average over N samples in batch
        loss = torch.mean(rmse_per_response)
        
        return loss