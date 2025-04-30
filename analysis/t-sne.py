import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

def plot_tsne(df: pd.DataFrame, target_col: str, n_components: int = 2, perplexity: float = 30.0, n_iter: int = 1000, name: Optional[str] = None):
    """
    Performs t-SNE on the features of the DataFrame and plots the components,
    colored by whether the target label is 'BENIGN' or not.

    Args:
        df: Input pandas DataFrame containing features and the target column.
        target_col: The name of the column containing the class labels.
        n_components: The number of components for t-SNE (usually 2 for visualization).
        perplexity: Related to the number of nearest neighbors considered (typical values 5-50).
        n_iter: Number of optimization iterations.
        name: Optional name to save the plot file (e.g., "tsne_plot"). If None, shows the plot.
    """
    # 1. Preprocessing
    # Handle potential infinite values and drop NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Create binary labels: BENIGN vs. MALICIOUS
    y_binary = y.apply(lambda label: 'BENIGN' if label == 'BENIGN' else 'MALICIOUS')

    # Ensure all feature columns are numeric, drop non-numeric ones if any exist
    X = X.select_dtypes(include=np.number)
    feature_names = X.columns # Store feature names after ensuring numeric types

    # Scale the features - t-SNE is sensitive to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. t-SNE
    # Note: t-SNE can be computationally expensive on large datasets.
    # Consider using a subset of data if it takes too long.
    print("Starting t-SNE... This might take a while.")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(X_scaled)
    print("t-SNE finished.")

    # Create a DataFrame for plotting
    tsne_df = pd.DataFrame(data=tsne_results,
                           columns=[f'TSNE{i+1}' for i in range(n_components)])
    # Use the binary labels for the hue column
    tsne_df['BENIGN_Status'] = y_binary.values

    # 3. Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='BENIGN_Status', # Use the binary column for hue
        data=tsne_df,
        palette='viridis', # You can choose other palettes
        alpha=0.7
    )

    plt.title(f't-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Pathogenicity') # Update legend title
    plt.grid(True)

    if name:
        plt.savefig(f"plots/{name}.png")
        print(f"t-SNE plot saved to plots/{name}.png")
    else:
        plt.show()

# Example Usage
if __name__ == '__main__':
    # Load your data
    data_path = "data/even_merged_1-10.csv"
    df = pd.read_csv(data_path).sample(n=5000, random_state=1729)

    # Specify the target column
    target_column = 'Label'

    # Generate and save the t-SNE plot
    # Adjust perplexity and n_iter based on dataset size and desired results
    plot_tsne(df, target_col=target_column, perplexity=50, n_iter=1000, name="t-sne")
