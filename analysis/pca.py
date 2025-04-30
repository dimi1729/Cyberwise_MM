import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

def plot_pca(df: pd.DataFrame, target_col: str, n_components: int = 2, name: Optional[str] = None):
    """
    Performs PCA on the features of the DataFrame and plots the first two principal components,
    colored by the target column.

    Args:
        df: Input pandas DataFrame containing features and the target column.
        target_col: The name of the column containing the class labels.
        n_components: The number of principal components to compute (default is 2).
        name: Optional name to save the plot file (e.g., "pca_plot"). If None, shows the plot.
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Create binary labels: BENIGN vs. MALICIOUS
    y_binary = y.apply(lambda label: 'BENIGN' if label == 'BENIGN' else 'MALICIOUS')

    # Ensure all feature columns are numeric, drop non-numeric ones if any exist
    X = X.select_dtypes(include=np.number)
    feature_names = X.columns # Store feature names after ensuring numeric types

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(data=principal_components,
                          columns=[f'PC{i+1}' for i in range(n_components)])
    # Use the binary labels for the hue column
    pca_df['BENIGN_Status'] = y_binary.values

    # 3. Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='BENIGN_Status', # Use the new binary column for hue
        data=pca_df,
        palette='viridis',
        alpha=0.7
    )

    plt.title(f'PCA of 39 features')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
    plt.legend(title='Pathogenicity')
    plt.grid(True)

    if name:
        plt.savefig(f"plots/{name}.png")
        print(f"PCA plot saved to plots/{name}.png")
    else:
        plt.show()

    print(f"Explained variance ratio by component: {pca.explained_variance_ratio_}")
    print(f"Total explained variance by {n_components} components: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

# Example Usage (assuming you run this script or import the function)
if __name__ == '__main__':
    # Load your data
    data_path = "data/even_merged_1-10.csv" # Or your specific data file
    df = pd.read_csv(data_path)

    # Specify the target column (e.g., 'Label')
    target_column = 'Label'

    # Generate and save the PCA plot
    plot_pca(df, target_col=target_column, name="pca")
