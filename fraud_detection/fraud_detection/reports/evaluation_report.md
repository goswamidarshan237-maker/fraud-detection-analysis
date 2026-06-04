# Fraud Detection Model Evaluation Report

## Overview
This report summarizes the performance of the trained Random Forest classifier on the test dataset.

## Summary Metrics
- **ROC-AUC Score:** 0.7574
- **PR-AUC (Average Precision) Score:** 0.0869

## Confusion Matrix
- **True Negatives (Legit identified as Legit):** 2898
- **False Positives (Legit flagged as Fraud):** 11
- **False Negatives (Fraud missed):** 89
- **True Positives (Fraud correctly flagged):** 2

## Classification Details
```
              precision    recall  f1-score   support

           0       0.97      1.00      0.98      2909
           1       0.15      0.02      0.04        91

    accuracy                           0.97      3000
   macro avg       0.56      0.51      0.51      3000
weighted avg       0.95      0.97      0.95      3000
```

## Performance Plots Saved
- [Confusion Matrix Plot](../visualizations/confusion_matrix.png)
- [ROC Curve](../visualizations/roc_curve.png)
- [Precision-Recall Curve](../visualizations/precision_recall_curve.png)
- [Feature Importance Plot](../visualizations/feature_importance.png)
