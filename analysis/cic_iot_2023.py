import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from plotting import plot_losses, plot_betas, plot_confusion_matrix

from typing import List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report

def split_cic_data(cic_data: pd.DataFrame,
                   train_frac: float,
                   val_frac: float,
                   test_frac: float) -> Tuple[torch.Tensor, ...]:
    # Split features and target
    #print(cic_data)
    cic_data = cic_data.replace([np.inf, -np.inf], np.nan).dropna()
    X = cic_data.drop(columns=['Label']).values
    y = (cic_data['Label'].values != "BENIGN").astype(float).reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # First split: training vs. temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_frac), random_state=42)
    
    if val_frac == 0:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_temp, dtype=torch.float32)
        y_test = torch.tensor(y_temp, dtype=torch.float32)
        return X_train, X_test, y_train, y_test 
    
    # Calculate proportion of validation data in the temp data
    val_ratio = val_frac / (val_frac + test_frac)

    
    # Second split: validation vs. test from temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, X_val, X_test, y_train, y_val, y_test 

class RegressionModel(nn.Module):
    def __init__(self, x, y):
        super(RegressionModel, self).__init__()
        self.features = x.shape[-1]
        self.linear = nn.Linear(self.features, 1)
        
    def forward(self, x):
        return self.linear(x)
    
    def loss(self, y_pred, y):
        return nn.functional.mse_loss(y_pred, y)



if __name__ == '__main__':
    df = pd.read_csv("data/Merged01.csv")

    x_train, x_val, x_test, y_train, y_val, y_test = split_cic_data(df, 0.7, 0.15, 0.15)

    model = RegressionModel(x_train, y_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    epochs = 3000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = model.loss(pred, y_train)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Training Loss: {loss.item():.4f}')
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = model.loss(val_pred, y_val)
                print(f'Epoch [{epoch}/{epochs}], Validation Loss: {val_loss.item():.4f}')
                val_losses.append(val_loss.item())
    
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test)
        test_loss = model.loss(test_pred, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')


    model_name: str = "lin_reg_epoch3000"    
    plot_losses(train_losses, val_losses, name=f'losses_{model_name}')
    plot_betas(model, name=f'betas_{model_name}')
    with torch.no_grad():
        test_pred = model(x_test)
        predicted_labels = (test_pred >= 0.5).float()

    cm = confusion_matrix(y_test.cpu().numpy(), predicted_labels.cpu().numpy())
    plot_confusion_matrix(cm, class_names=["Malicious", "Benign"], name="cm_lin_reg_epoch3000")