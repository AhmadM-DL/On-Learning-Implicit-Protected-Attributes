import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

multilabel = False
metrics = {}

test_predict_probability= pd.read_csv("/content/On-Learning-Implicit-Protected-Attributes/test_predict_probability.csv", index_col=0)
labels = [c for c in test_predict_probability if not "_gt" in c]

predictions = test_predict_probability[ [c for c in test_predict_probability if not "_gt" in c]].values
gt = test_predict_probability[ [c for c in test_predict_probability if "_gt" in c]].values

if multilabel == True:
  roc_auc_scores= {}
  roc_auc_curves= {}
  for i, (prediction, ground_truth) in enumerate(zip(predictions.T, gt.T)):
    roc_auc_scores[labels[i]] = roc_auc_score(ground_truth, prediction)
    fpr, tpr, thresholds = roc_curve(ground_truth, prediction)
    roc_auc_curves[labels[i]] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
  metrics["roc_auc_scores"] = roc_auc_scores
  metrics["roc_auc_curves"] = roc_auc_curves
else:
  predictions = [labels[p] for p in np.argmax(predictions, axis=1)]
  gt = [labels[g] for g in np.argmax(gt, axis=1)]
  metrics["accuracy_score"] = accuracy_score(predictions, gt)
  metrics["confusion_matrix"] = confusion_matrix(predictions, gt, normalize="true", labels=labels)
metrics