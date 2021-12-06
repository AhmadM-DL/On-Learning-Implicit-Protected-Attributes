import pandas as pd
import numpy as np
import json, os
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from predict import predict

def eval(dataset, split_path, model_name, pretrain_model_path, output_dir, multilabel,
         batch_size= 32, class_mode= "raw", height=320, width=320):
  metrics = {}
  predictions, gt, labels = predict(dataset, split_path, model_name, pretrain_model_path, output_dir=output_dir,
                            tag="test_predict_probability", batch_size= batch_size, class_mode= class_mode, 
                            height=height, width=width)
  if multilabel == True:
    roc_auc_scores= {}
    roc_auc_curves= {}
    for i, (prediction, ground_truth) in enumerate(zip(predictions.T, gt.T)):
      roc_auc_scores[labels[i]] = roc_auc_score(ground_truth, prediction)
      fpr, tpr, thresholds = roc_curve(ground_truth, prediction)
      roc_auc_curves[labels[i]] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    metrics["roc_auc_scores"] = roc_auc_scores
    metrics["roc_auc_curves"] = roc_auc_curves
    metrics["labels"] = labels
  else:
    predictions = [labels[p] for p in np.argmax(predictions, axis=1)]
    gt = [labels[g] for g in np.argmax(gt, axis=1)]
    metrics["accuracy_score"] = accuracy_score(predictions, gt)
    metrics["confusion_matrix"] = confusion_matrix(predictions, gt, normalize="true", labels=labels).tolist()
    metrics["labels"] = labels
  if output_dir:
    json.dump(metrics, open(os.path.join(output_dir, "evaluation.json"), "w"))
  return metrics