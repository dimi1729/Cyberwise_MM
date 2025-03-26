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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Entire plot
    ax1.step(recall, precision, where='post', label='Precision-Recall Curve')
    ax1.set_xlim(0, 1.01)
    ax1.set_ylim(0, 1.01)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend()
    ax1.grid(False)

    # Zoomed in plot
    ax2.step(recall, precision, where='post', label='Precision-Recall Curve')
    ax2.set_xlim(0.95, 1.001)
    ax2.set_ylim(0.95, 1.001)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Zoomed Precision-Recall Curve')
    ax2.legend()
    ax2.grid(False)

    if name is not None:
        plt.savefig(f"plots/{name}.png")
    else:
        plt.show()

def plot_multiple_precision_recall_curves(precisions: List[List[float]], recalls: List[List[float]], labels: List[str], name: Optional[str] = None):
    if not (len(precisions) == len(recalls) == len(labels)):
        raise ValueError("The lengths of precisions, recalls, and labels must be equal.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for precision, recall, label in zip(precisions, recalls, labels):
        ax1.step(recall, precision, where='post', label=label)
        ax2.step(recall, precision, where='post', label=label)

    ax1.set_xlim(0, 1.01)
    ax1.set_ylim(0, 1.01)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves for Multiple Models')
    ax1.legend(loc='lower left')
    ax1.grid(False)

    ax2.set_xlim(0.95, 1.001)
    ax2.set_ylim(0.95, 1.001)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Zoomed Precision-Recall Curves')
    ax2.legend(loc='lower left')
    ax2.grid(False)
    
    if name is not None:
        plt.savefig(f"plots/{name}.png")
    else:
        plt.show()

def plot_roc_curve(fpr: List[float], tpr: List[float], roc_auc: float, name: Optional[str] = None):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random Chance')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Full ROC curve with Random Chance line
    ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random Chance')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    # Zoomed ROC curve (top left area) without Random Chance line
    ax2.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Zoomed ROC Curve')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    ax2.set_xlim(0, 0.25)
    ax2.set_ylim(0.95, 1)
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    if name is not None:
        plt.savefig(f"plots/{name}.png")
    else:
        plt.show()


def plot_multiple_roc_curves(fprs: List[List[float]], tprs: List[List[float]], roc_aucs: List[float], labels: List[str], name: Optional[str] = None):
    if not (len(fprs) == len(tprs) == len(roc_aucs) == len(labels)):
        raise ValueError("The lengths of fprs, tprs, roc_aucs, and labels must be equal.")

    plt.figure(figsize=(10, 6))
    for fpr, tpr, roc_auc, label in zip(fprs, tprs, roc_aucs, labels):
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
    # Create subplots for full and zoomed ROC curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Plot each ROC curve on both axes
    for fpr, tpr, roc_auc, label in zip(fprs, tprs, roc_aucs, labels):
        ax1.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
        ax2.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
    # Full ROC curve with baseline on ax1
    ax1.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random Chance')
    ax1.set_xlim(0, 1.01)
    ax1.set_ylim(0, 1.01)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    # Zoomed ROC curves on ax2
    ax2.set_xlim(0, 0.25)
    ax2.set_ylim(0.95, 1)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Zoomed ROC Curves')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiple ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    if name is not None:
        plt.savefig(f"plots/{name}.png")
    else:
        plt.show()