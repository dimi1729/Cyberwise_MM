import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_curve

from typing import List, Tuple, Optional

from analysis.preprocessing import split_cic_data
import analysis.plotting as plotting
from models.linear_regression import RegressionModel
from models.decision_trees import RandomForestAnalyzer, XGBoostRFAnalyzer


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
        loss = model.weighted_mse_loss(pred, y_train, (1, 1))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        if epoch % 100 == 0:
            # print(f'Epoch [{epoch}/{epochs}], Training Loss: {loss.item():.4f}')
            # train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                train_mse_loss = model.mse_loss(pred, y_train)
                train_losses.append(train_mse_loss.item())
                val_pred = model(x_val)
                val_loss = model.mse_loss(val_pred, y_val)
                print(f'Epoch [{epoch}/{epochs}], Training Loss: {train_mse_loss.item():.4f} Validation Loss: {val_loss.item():.4f}')
                val_losses.append(val_loss.item())
    
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test)
        test_loss = model.mse_loss(test_pred, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')


    model_name: str = "lin_reg_epoch3000"    
    # plot_losses(train_losses, val_losses, name=f'losses_{model_name}')
    #plot_betas(model, name=f'betas_{model_name}')
    with torch.no_grad():
        test_pred = model(x_test)
        predicted_labels = (test_pred >= 0.5).float()
    cm = confusion_matrix(y_test.cpu().numpy(), predicted_labels.cpu().numpy())

    with torch.no_grad():
        test_pred = model(x_test)
        test_scores = test_pred.cpu().numpy()
        y_true = y_test.cpu().numpy()
    
    output = {
        "predictions": test_scores.tolist(),
        "actual": y_true.tolist()
    }
    with open(f"results/{model_name}.json", "w") as f:
        json.dump(output, f)
    
    quit()

    precision, recall, thresholds = precision_recall_curve(y_true, test_scores)
    plotting.plot_precision_recall_curve(precision, recall)
    plotting.plot_betas(model, name=f'betas_{model_name}')