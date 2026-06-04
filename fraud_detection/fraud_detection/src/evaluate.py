import os
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for headless environments (no display needed)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import joblib

def evaluate_model(model_path, processed_dir, visualizations_dir, reports_dir):
    """
    Loads test data and trained model, generates classifications,
    plots evaluation curves, and outputs an assessment report.
    """
    print("Loading test data and model...")
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.ravel()
    
    model = joblib.load(model_path)
    
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability of fraud class (1)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    print("\nTest Classification Report:")
    print(report_text)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Create output directories
    os.makedirs(visualizations_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], 
                yticklabels=['Legit', 'Fraud'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    # 2. Plot ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, "roc_curve.png"), dpi=150)
    plt.close()
    
    # 3. Plot Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, "precision_recall_curve.png"), dpi=150)
    plt.close()
    
    # 4. Plot Feature Importance
    importances = model.feature_importances_
    features = X_test.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=True) # Ascending for horizontal bar plot
    
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
    plt.xlabel('Relative Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, "feature_importance.png"), dpi=150)
    plt.close()
    
    # 5. Save Evaluation Report Markdown
    report_md_path = os.path.join(reports_dir, "evaluation_report.md")
    with open(report_md_path, "w") as f:
        f.write("# Fraud Detection Model Evaluation Report\n\n")
        f.write("## Overview\n")
        f.write("This report summarizes the performance of the trained Random Forest classifier on the test dataset.\n\n")
        
        f.write("## Summary Metrics\n")
        f.write(f"- **ROC-AUC Score:** {roc_auc:.4f}\n")
        f.write(f"- **PR-AUC (Average Precision) Score:** {pr_auc:.4f}\n\n")
        
        f.write("## Confusion Matrix\n")
        f.write(f"- **True Negatives (Legit identified as Legit):** {cm[0, 0]}\n")
        f.write(f"- **False Positives (Legit flagged as Fraud):** {cm[0, 1]}\n")
        f.write(f"- **False Negatives (Fraud missed):** {cm[1, 0]}\n")
        f.write(f"- **True Positives (Fraud correctly flagged):** {cm[1, 1]}\n\n")
        
        f.write("## Classification Details\n")
        f.write("```\n")
        f.write(report_text)
        f.write("```\n\n")
        
        f.write("## Performance Plots Saved\n")
        f.write("- [Confusion Matrix Plot](../visualizations/confusion_matrix.png)\n")
        f.write("- [ROC Curve](../visualizations/roc_curve.png)\n")
        f.write("- [Precision-Recall Curve](../visualizations/precision_recall_curve.png)\n")
        f.write("- [Feature Importance Plot](../visualizations/feature_importance.png)\n")
        
    print(f"Evaluation report written to {report_md_path}")
    print(f"Visualizations saved to {visualizations_dir}")

if __name__ == "__main__":
    model_output_path = os.path.join("models", "fraud_detector_model.joblib")
    processed_data_dir = os.path.join("data", "processed")
    viz_dir = os.path.join("visualizations")
    rep_dir = os.path.join("reports")
    
    if os.path.exists(model_output_path) and os.path.exists(os.path.join(processed_data_dir, "X_test.csv")):
        evaluate_model(model_output_path, processed_data_dir, viz_dir, rep_dir)
    else:
        print("Required model or test datasets are missing.")
