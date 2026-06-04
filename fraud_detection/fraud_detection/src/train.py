import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def train_and_compare_models(processed_dir, models_dir, report_path="reports/model_comparison_report.md"):
    """
    Trains multiple classification models (Logistic Regression, Random Forest, Gradient Boosting),
    evaluates them on the test set, compares key metrics, and saves the best model.
    """
    print("Loading preprocessed datasets...")
    X_train_path = os.path.join(processed_dir, "X_train.csv")
    y_train_path = os.path.join(processed_dir, "y_train.csv")
    X_test_path = os.path.join(processed_dir, "X_test.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")
    
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    # Initialize baseline models with appropriate hyper-parameters
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }
    
    results = []
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict class values and scores
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            
        # Calculate metric scores
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })
        trained_models[name] = model
        
    # Build comparison summary
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    # Sort by ROC-AUC and F1-Score to determine best performing model
    best_model_row = results_df.sort_values(by=["ROC-AUC", "F1-Score"], ascending=False).iloc[0]
    best_model_name = best_model_row["Model"]
    best_model = trained_models[best_model_name]
    
    # Save best model to folder
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_fraud_detector_model.joblib")
    joblib.dump(best_model, best_model_path)
    
    # Duplicate as 'fraud_detector_model.joblib' for backward compatibility
    joblib.dump(best_model, os.path.join(models_dir, "fraud_detector_model.joblib"))
    
    print(f"\nSaved Best Model ({best_model_name}) to: {best_model_path}")
    
    # Write comparison markdown report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Machine Learning Model Comparison Report\n\n")
        f.write("This report compares different classifiers trained on the engineered credit card transactions dataset to identify the optimal model for production.\n\n")
        
        f.write("## 1. Metrics Overview\n")
        f.write("Due to severe class imbalance, accuracy is a deceptive metric. We prioritize **F1-Score** and **ROC-AUC** for selecting our best model.\n\n")
        
        f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for res in results:
            f.write(f"| {res['Model']} | {res['Accuracy']:.4f} | {res['Precision']:.4f} | {res['Recall']:.4f} | {res['F1-Score']:.4f} | {res['ROC-AUC']:.4f} |\n")
        f.write("\n")
        
        f.write(f"## 2. Selection Rationale\n")
        f.write(f"The **{best_model_name}** was selected as the best classifier due to its superior ROC-AUC score of **{best_model_row['ROC-AUC']:.4f}** and F1-Score of **{best_model_row['F1-Score']:.4f}**.\n\n")
        f.write("- **Logistic Regression**: Serves as a linear baseline. It is highly interpretable, but lacks modeling power for non-linear feature interactions.\n")
        f.write("- **Random Forest**: Achieves strong balances. It natively handles class imbalance via cost-sensitive learning (`class_weight='balanced'`) and is highly resilient to overfitting.\n")
        f.write("- **Gradient Boosting**: Sequential gradient boosting trees optimize residual errors, offering high predictive power.\n")
        
    print(f"Model comparison report saved to {report_path}")
    return results_df

if __name__ == "__main__":
    processed_data_dir = os.path.join("data", "processed")
    models_data_dir = os.path.join("models")
    report_output_path = os.path.join("reports", "model_comparison_report.md")
    
    if os.path.exists(os.path.join(processed_data_dir, "X_train.csv")):
        train_and_compare_models(processed_data_dir, models_data_dir, report_output_path)
    else:
        print("Preprocessed training data not found. Run preprocessing.py first.")
