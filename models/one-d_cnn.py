import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import json
from analysis.preprocessing import split_cic_data
from typing import Tuple

class OneDCNN(nn.Module):
    def __init__(self, n_features: int, conv_out_channels: int = 16, kernel_size: int = 3,
                 fc_hidden_size: int = 32, dropout: float = 0.5):
        """
        n_features: number of input features per sample.
        conv_out_channels: number of output channels (filters) for the Conv1d layer.
        kernel_size: kernel size for the Conv1d layer.
        fc_hidden_size: number of units for the first fully connected layer.
        dropout: dropout probability.
        """
        super(OneDCNN, self).__init__()
        # For a 1D CNN, treat the features as a 1D signal with one channel.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_out_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv_out_channels, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: input tensor of shape [batch_size, n_features]
        returns: predictions of shape [batch_size, 1]
        """
        # Unsqueeze to get shape [batch_size, 1, n_features]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) 
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc1(x)  
        x = self.relu(x)
        x = self.fc2(x)  
        return x

    def mse_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_pred, y)
    
    def weighted_mse_loss(self, y_pred: torch.Tensor, y: torch.Tensor, weights: Tuple[float, float]) -> torch.Tensor:
        pos_class_weight, neg_class_weight = weights
        diff = y_pred - y
        squared_error = diff ** 2
        weights_tensor = torch.where(y == 1.0, pos_class_weight, neg_class_weight)
        loss = torch.mean(weights_tensor * squared_error)
        return loss

if __name__ == '__main__':

    df = pd.read_csv("data/Merged01.csv")
    x_train, x_val, x_test, y_train, y_val, y_test = split_cic_data(df, 0.7, 0.15, 0.15)
    
    n_features = x_train.shape[1]
    model = OneDCNN(n_features)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = model.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item():.4f}")
    
    print(f"Training Loss: {loss.item():.4f}")
    
    # Evaluate on validation and test sets
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = model.mse_loss(val_pred, y_val)
        print(f"Validation Loss: {val_loss.item():.4f}")
        
        test_pred = model(x_test)
        test_loss = model.mse_loss(test_pred, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")
    
    # Optionally, save results to JSON
    output = {
        "val_predictions": val_pred.cpu().numpy().tolist(),
        "val_actual": y_val.cpu().numpy().tolist(),
        "test_predictions": test_pred.cpu().numpy().tolist(),
        "test_actual": y_test.cpu().numpy().tolist()
    }
    with open("results/one_d_cnn_results.json", "w") as f:
        json.dump(output, f)