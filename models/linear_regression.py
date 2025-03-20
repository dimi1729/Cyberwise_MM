import torch
import torch.nn as nn

from typing import Tuple

class RegressionModel(nn.Module):
    def __init__(self, x, y):
        super(RegressionModel, self).__init__()
        self.features = x.shape[-1]
        self.linear = nn.Linear(self.features, 1)
        
    def forward(self, x):
        return self.linear(x)
    
    def mse_loss(self, y_pred, y):
        return nn.functional.mse_loss(y_pred, y)
    
    def weighted_mse_loss(self, y_pred, y, weights: Tuple[float, float]):
        pos_class_weight, neg_class_weight = weights
        diff = y_pred - y
        squared_error = diff ** 2
        weights_tensor = torch.where(y == 1.0, pos_class_weight, neg_class_weight)
        loss = torch.mean(weights_tensor * squared_error)
        return loss
    
    def lasso_mse_loss(self, y_pred, y, lam: float):
        mse_loss = nn.functional.mse_loss(y_pred, y)
        l1_penalty = lam * torch.sum(torch.abs(self.linear.weight))
        return mse_loss + l1_penalty
    
    def ridge_mse_loss(self, y_pred, y, lam: float):
        mse_loss = nn.functional.mse_loss(y_pred, y)
        l2_penalty = lam * torch.sum(self.linear.weight ** 2)
        return mse_loss + l2_penalty