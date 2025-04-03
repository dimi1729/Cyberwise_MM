import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json

from typing import Tuple

from analysis.preprocessing import split_cic_data

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


if __name__ == '__main__':

    df = pd.read_csv("data/realistic_merged_1-10.csv") 
    x_train_np, x_val_np, x_test_np, y_train_np, y_val_np, y_test_np = split_cic_data(df, 0.7, 0.15, 0.15)
    
    # Convert numpy arrays to torch tensors
    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    
    # Instantiate the regression model
    model = RegressionModel(x_train, y_train)
    
    # Define an optimizer (here we use Adam) and learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3000
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = model.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = model.mse_loss(model(x_train), y_train)
                val_loss = model.mse_loss(model(x_val), y_val)
                print(f"Epoch [{epoch}/{epochs}], Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
    
    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test)
        test_loss = model.mse_loss(test_pred, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")
    
    # Optionally, save test predictions and actual values as JSON
    output = {
        "test_predictions": test_pred.cpu().numpy().tolist(),
        "test_actual": y_test.cpu().numpy().tolist()
    }
    with open("results/ling_reg_e3000_realistic.json", "w") as f:
        json.dump(output, f)