import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import List, Tuple, Optional

from typing import Tuple


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

