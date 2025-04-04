import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from typing import Dict, List, Any, Optional
from sklearn.metrics import roc_curve, roc_auc_score

import analysis.plotting as plotting

def plot_rocs_and_prcs(results: Dict[str, Dict], combined_name: Optional[str] = None) -> None:
    precisions = []
    recalls = []
    fprs = []
    tprs = []
    roc_aucs = []
    for name, json_dict in results.items():
        test_scores = json_dict["test_predictions"]
        y_true = json_dict["test_actual"]

        precision, recall, thresholds = precision_recall_curve(y_true, test_scores)
        precisions.append(precision)
        recalls.append(recall)

        fpr, tpr, _ = roc_curve(y_true, test_scores)
        roc_auc = roc_auc_score(y_true, test_scores)
        #plotting.plot_roc_curve(fpr, tpr, roc_auc)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
        plotting.plot_precision_recall_curve(precision, recall, name=f"prc_{name}")
        plotting.plot_roc_curve(fpr, tpr, roc_auc, name=f"roc_{name}")
    
    if combined_name is not None:
        plotting.plot_multiple_precision_recall_curves(precisions, recalls,
                                                    labels = list(results.keys()),
                                                    name=f"prcs_{name}")
        plotting.plot_multiple_roc_curves(fprs, tprs, roc_aucs,
                                        labels = list(results.keys()),
                                        name=f"rocs_{name}")



if __name__ == '__main__':
    
    with open("results/oned_cnn_3000e.json", 'r') as f:
        lin_reg_results: Dict[str, List[List[float]]] = json.load(f)

    #plot_rocs_and_prcs({"lin_reg_e3000_1-1": lin_reg_results})
    y_true = [item[0] for item in lin_reg_results["test_actual"]]
    #print(y_true)
    y_pred = [1 if score[0] >= 0.5 else 0 for score in lin_reg_results["test_predictions"]]
    #print(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    plotting.plot_confusion_matrix(cm, class_names=["Benign", "Malicious"], name="cm_e3000_1-1")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

        
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')