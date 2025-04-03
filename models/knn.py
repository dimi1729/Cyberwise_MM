import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from analysis.preprocessing import split_cic_data


class KNNModel(nn.Module):
    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, k: int = 5):
        """
        x_train: training features, tensor of shape [n_train, n_features]
        y_train: training targets, tensor of shape [n_train, 1]
        k: number of nearest neighbors to use
        """
        super(KNNModel, self).__init__()
        self.k = k
        self.register_buffer('x_train', x_train)
        self.register_buffer('y_train', y_train)

    def forward(self, x: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
        """
        For each sample in x, compute the Euclidean distance to all training examples, 
        then average the targets of the k nearest ones.
        
        x: input features, shape [batch_size, n_features]
        batch_size: number of samples to process in one chunk to avoid memory issues.
        returns: predictions, shape [num_samples, 1]
        """
        preds = []
        # Process x in batches to avoid huge temporary tensors.
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i: i + batch_size]
            distances = torch.cdist(x_batch, self.x_train)
            _, indices = distances.topk(k=self.k, largest=False)
            # Gather corresponding training targets: shape [batch_size, k, 1]
            knn_targets = self.y_train[indices]
            # Average along neighbor dimension
            pred_batch = torch.mean(knn_targets, dim=1)
            preds.append(pred_batch)
            print('finished batch', i)
        return torch.cat(preds, dim=0)

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
    # Load dataset
    df = pd.read_csv("data/even_merged_1-10.csv")
    
    x_train, x_val, x_test, y_train, y_val, y_test = split_cic_data(df, 0.7, 0.15, 0.15)
    
    model = KNNModel(x_train, y_train, k=5)
    
    model.eval()
    with torch.no_grad():
        # Validation predictions and loss
        val_pred = model(x_val)
        val_loss = model.mse_loss(val_pred, y_val)
        print(f'Validation Loss: {val_loss.item():.4f}')
        
        # Test predictions and loss
        test_pred = model(x_test)
        test_loss = model.mse_loss(test_pred, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
    
    output = {
        "val_predictions": val_pred.cpu().numpy().tolist(),
        "val_actual": y_val.cpu().numpy().tolist(),
        "test_predictions": test_pred.cpu().numpy().tolist(),
        "test_actual": y_test.cpu().numpy().tolist()
    }
    with open("results/knn_results.json", "w") as f:
        json.dump(output, f)