import torch
from torch import nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        if yhat.is_complex() or y.is_complex():
            diff = yhat - y
            mse = torch.mean(diff.real**2 + diff.imag**2)
        else:
            mse = self.mse(yhat, y)

        loss = torch.sqrt(mse + self.eps)
        return loss
