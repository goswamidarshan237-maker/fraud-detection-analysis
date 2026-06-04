import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data(raw_path, processed_dir, models_dir=None, report_path="reports/preprocessing_report.md"):
    """
    Loads raw data, performs data cleaning (missing value imputation, duplicate removal, 
    outlier handling), encodes categorical features, scales numerical features,
    and saves split datasets and a comparison report.
    """
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Store initial metrics for comparison
    initial_shape = df.shape
    initial_missing = df.isnull().sum().sum()
    initial_duplicates = df.duplicated().sum()
    
    # Outlier counts before processing using the Interquartile Range (IQR) method
    num_cols = ['age', 'hour', 'day_of_week', 'amount', 'distance_from_home', 'transactions_per_user', 'avg_transaction_amount', 'time_since_last_tx', 'tx_velocity_1h']
    cat_cols = ['category', 'device', 'amount_category']
    
    outliers_before = {}
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers_before[col] = outliers_count

    # 1. Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_shape[0] - df_cleaned.shape[0]
    
    # 2. Impute missing values
    # Standard SimpleImputer using median for numerical and most frequent for categorical features
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    df_cleaned[num_cols] = num_imputer.fit_transform(df_cleaned[num_cols])
    df_cleaned[cat_cols] = cat_imputer.fit_transform(df_cleaned[cat_cols])
    
    final_missing = df_cleaned.isnull().sum().sum()
    
    # 3. Handle outliers
    # In fraud detection, extreme amounts or distances are highly informative. Rather than dropping them,
    # we clip values at the 99.5th percentile to prevent high variance from distorting the standard scale.
    clipping_thresholds = {}
    for col in ['amount', 'distance_from_home']:
        upper_limit = df_cleaned[col].quantile(0.995)
        clipping_thresholds[col] = upper_limit
        df_cleaned[col] = np.clip(df_cleaned[col], 0, upper_limit)
        
    outliers_after = {}
    for col in num_cols:
        q1 = df_cleaned[col].quantile(0.25)
        q3 = df_cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
        outliers_after[col] = outliers_count

    # Separate identifiers, target, and features
    X = df_cleaned.drop(columns=['transaction_id', 'user_id', 'timestamp', 'is_fraud'], errors='ignore')
    y = df_cleaned['is_fraud']
    
    # Stratified train-test split to preserve imbalance class distributions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Standard Scaling of Numerical Features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])
    
    # 5. One-Hot Encoding of Categorical Features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])
    
    # Column metadata extraction
    cat_feature_names = list(encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    
    # Reassemble complete feature matrices
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])
    
    # Convert into standard Pandas DataFrames
    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)
    
    # Write files to processed directory
    os.makedirs(processed_dir, exist_ok=True)
    X_train_df.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    # Save the pipeline artifacts
    if models_dir:
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
        joblib.dump(encoder, os.path.join(models_dir, "encoder.joblib"))
        joblib.dump(num_imputer, os.path.join(models_dir, "num_imputer.joblib"))
        joblib.dump(cat_imputer, os.path.join(models_dir, "cat_imputer.joblib"))
        print(f"Scaler, Encoder, and Imputers successfully saved to {models_dir}")
        
    # 6. Generate Preprocessing Comparison Report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Data Preprocessing & Cleaning Report\n\n")
        f.write("This report provides a comparison of the dataset before and after the preprocessing pipeline steps (duplicate removal, missing value imputation, outlier handling, categorical encoding, and feature scaling).\n\n")
        
        f.write("## 1. General Cleaning Metrics\n")
        f.write("| Metric | Before | After | Change / Detail |\n")
        f.write("| --- | --- | --- | --- |\n")
        f.write(f"| **Dataset Shape** | {initial_shape} | {df_cleaned.shape} | Removed {duplicates_removed} duplicates |\n")
        f.write(f"| **Missing Values** | {initial_missing} | {final_missing} | Imputed using Median/Most Frequent strategy |\n")
        f.write(f"| **Duplicate Rows** | {initial_duplicates} | 0 | Cleaned |\n\n")
        
        f.write("## 2. Outlier Detection Summary (IQR Method)\n")
        f.write("Extreme transactional attributes often correlate with fraudulent activity. To limit extreme variance without losing these patterns, variables were capped at their 99.5th percentile prior to scaling.\n\n")
        f.write("| Feature | Outliers Before (IQR) | Outliers After (IQR) | Upper Cap Limit (99.5th percentile) |\n")
        f.write("| --- | --- | --- | --- |\n")
        for col in num_cols:
            cap_val = f"{clipping_thresholds[col]:.2f}" if col in clipping_thresholds else "N/A (No cap)"
            f.write(f"| {col} | {outliers_before[col]} | {outliers_after[col]} | {cap_val} |\n")
        f.write("\n")
        
        f.write("## 3. Categorical Feature Encoding\n")
        f.write("Categorical elements were transformed into binary features using One-Hot Encoding:\n")
        f.write(f"- **Original features:** {cat_cols}\n")
        f.write(f"- **One-Hot Encoded features:** {cat_feature_names}\n\n")
        
        f.write("## 4. Scaled Feature Verification (Train Split)\n")
        f.write("Summary statistics after Standard Scaling (should have mean ~ 0, std ~ 1):\n")
        f.write("```\n")
        f.write(X_train_df[num_cols].describe().to_string())
        f.write("\n```\n")
        
    print(f"Preprocessing report written to {report_path}")
    print(f"Data successfully split and saved to {processed_dir}")
    return X_train_df, X_test_df, y_train, y_test

if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "transactions_engineered.csv")
    if not os.path.exists(raw_data_path):
        raw_data_path = os.path.join("data", "raw", "transactions.csv")
    processed_data_dir = os.path.join("data", "processed")
    models_data_dir = os.path.join("models")
    
    if os.path.exists(raw_data_path):
        preprocess_data(raw_data_path, processed_data_dir, models_data_dir)
    else:
        print("Raw data file not found. Run src/data_generator.py first.")
