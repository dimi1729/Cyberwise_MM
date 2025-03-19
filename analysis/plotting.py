import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional
import numpy as np

def plot_losses(train_loss: List[float], val_loss: List[float], name: Optional[str] = None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    if name is not None:
        plt.savefig(f"plots/{name}.png")
    else:
        plt.show()

def plot_confusion_matrix(cm, class_names: Optional[List[str]] = None, name: Optional[str] = None):
    # If no class names are provided, generate default names
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    if name is not None:
        plt.savefig(f"plots/{name}_confusion_matrix.png")
    else:
        plt.show()


def plot_betas(model, name: Optional[str] = None):
    betas = model.linear.weight.detach().cpu().numpy().flatten()
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(betas)), betas)
    plt.xlabel('Feature Index')
    plt.ylabel('Beta Value')
    plt.title('Beta Coefficients from the Linear Regression Model')
    plt.grid(True)
    if name is not None:
        plt.savefig(f"plots/{name}_betas.png")
    else:
        plt.show()