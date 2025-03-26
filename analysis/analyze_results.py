import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_curve

from typing import Dict, List
from sklearn.metrics import roc_curve, roc_auc_score

import analysis.plotting as plotting

def plot_rocs_and_prcs():
    precisions = []
    recalls = []
    fprs = []
    tprs = []
    roc_aucs = []
    for result in [pos_1x, pos_5x, pos_10x, pos_20x]:
        test_scores = result["predictions"]
        y_true = result["actual"]

        precision, recall, thresholds = precision_recall_curve(y_true, test_scores)
        precisions.append(precision)
        recalls.append(recall)

        fpr, tpr, _ = roc_curve(y_true, test_scores)
        roc_auc = roc_auc_score(y_true, test_scores)
        #plotting.plot_roc_curve(fpr, tpr, roc_auc)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
    plotting.plot_multiple_precision_recall_curves(precisions, recalls,
                                                   labels = ["No weight", "5x", "10x", "20x"],
                                                   name="linear_prcs_epoch3000")
    plotting.plot_multiple_roc_curves(fprs, tprs, roc_aucs,
                                      labels = ["No weight", "5x", "10x", "20x"],
                                      name='linear_rocs_epoch3000')



if __name__ == '__main__':
    
    with open("results/lin_reg_epoch3000.json", 'r') as f:
        pos_1x: Dict[str, List[List[float]]] = json.load(f)
    with open("results/lin_reg_pos5x_epoch3000.json", 'r') as f:
        pos_5x: Dict[str, List[List[float]]] = json.load(f)   
    with open("results/lin_reg_pos10x_epoch3000.json", 'r') as f:
        pos_10x: Dict[str, List[List[float]]] = json.load(f)   
    with open("results/lin_reg_pos20x_epoch3000.json", 'r') as f:
        pos_20x: Dict[str, List[List[float]]] = json.load(f)   
    
    model = torch.load("results/lin_reg_epoch3000_model.pt")
    plotting.plot_betas(model, name="linear_betas_epoch3000")

 