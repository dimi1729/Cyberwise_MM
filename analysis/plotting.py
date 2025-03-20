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
        plt.savefig(f"plots/{name}.png")
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
        plt.savefig(f"plots/{name}.png")
    else:
        plt.show()

def plot_feature_importance(importances: np.ndarray, 
                          feature_names: List[str], 
                          name: str,
                          top_n: int = 20):
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[-top_n:]
    plt.title(f'Feature Importance - {name}')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'plots/{name}.png')
    else:
        plt.show()
 
def plot_precision_recall_curve(precision: List[float], recall: List[float], name: Optional[str] = None):
    plt.figure(figsize=(10, 6))
    plt.step(recall, precision, where='post', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    if name is not None:
        plt.savefig(f"plots/{name}.png")
    else:
        plt.show()