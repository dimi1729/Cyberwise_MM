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

def merge_cic_csvs(csv_paths: List[str],
                   benign_to_malicious_ratio: Tuple[float, float],
                   output_path: Optional[str] = None):
    """
    Merge CSV files and sample benign and malicious records according to the specified ratio.
    
    Args:
        csv_paths: List of paths to CSV files
        benign_to_malicious_ratio: Tuple of (benign_ratio, malicious_ratio)
        output_path: Optional path to save the merged CSV
        
    Returns:
        pd.DataFrame: The merged and sampled dataframe
    """
    # Merge all CSVs
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Separate benign and malicious records
    benign_df = merged_df[merged_df['Label'] == "BENIGN"]
    malicious_df = merged_df[merged_df['Label'] != "BENIGN"]
    
    benign_count = len(benign_df)
    malicious_count = len(malicious_df)
    print(f"Total records: {len(merged_df)} (Benign: {benign_count}, Malicious: {malicious_count})")
    
    # Check if we have both classes
    if benign_count == 0 or malicious_count == 0:
        raise ValueError("Both benign and malicious samples must be present in the data")
    
    # Calculate how many samples to take from each class
    benign_ratio, malicious_ratio = benign_to_malicious_ratio
    
    # Determine limiting class based on ratio
    if (benign_count / benign_ratio) < (malicious_count / malicious_ratio):
        # Benign is limiting
        target_benign = benign_count
        target_malicious = int(target_benign * (malicious_ratio / benign_ratio))
    else:
        # Malicious is limiting
        target_malicious = malicious_count
        target_benign = int(target_malicious * (benign_ratio / malicious_ratio))
    
    # Sample from each class
    sampled_benign = benign_df.sample(n=target_benign, random_state=42)
    sampled_malicious = malicious_df.sample(n=target_malicious, random_state=42)
    
    # Combine and shuffle
    result_df = pd.concat([sampled_benign, sampled_malicious], ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV if output path is provided
    if output_path:
        print(f"Saving {len(result_df)} records to {output_path}")
        result_df.to_csv(output_path, index=False)
    
    return result_df

if __name__ == '__main__':
    cic_csvs = ["data/Merged01.csv", "data/Merged02.csv", "data/Merged03.csv", 
                "data/Merged04.csv", "data/Merged05.csv", "data/Merged06.csv", 
                "data/Merged07.csv", "data/Merged08.csv", "data/Merged09.csv", 
                "data/Merged10.csv", 
                ]
    # merge_cic_csvs(cic_csvs, (1, 1), "data/even_merged_1-10.csv")
    merge_cic_csvs(cic_csvs, (100, 1), "data/100:1_merged_1-10.csv")