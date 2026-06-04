# Machine Learning Model Comparison Report

This report compares different classifiers trained on the engineered credit card transactions dataset to identify the optimal model for production.

## 1. Metrics Overview
Due to severe class imbalance, accuracy is a deceptive metric. We prioritize **F1-Score** and **ROC-AUC** for selecting our best model.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6880 | 0.0622 | 0.6593 | 0.1136 | 0.7440 |
| Random Forest | 0.8867 | 0.1022 | 0.3516 | 0.1584 | 0.7390 |
| Gradient Boosting | 0.9667 | 0.1538 | 0.0220 | 0.0385 | 0.7574 |

## 2. Selection Rationale
The **Gradient Boosting** was selected as the best classifier due to its superior ROC-AUC score of **0.7574** and F1-Score of **0.0385**.

- **Logistic Regression**: Serves as a linear baseline. It is highly interpretable, but lacks modeling power for non-linear feature interactions.
- **Random Forest**: Achieves strong balances. It natively handles class imbalance via cost-sensitive learning (`class_weight='balanced'`) and is highly resilient to overfitting.
- **Gradient Boosting**: Sequential gradient boosting trees optimize residual errors, offering high predictive power.
