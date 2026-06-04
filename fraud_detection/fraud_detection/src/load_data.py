import os
import pandas as pd

def load_and_profile_data(csv_path="data/raw/transactions.csv", report_path="reports/data_profile_report.md"):
    """
    Loads a fraud transactions CSV, prints basic profiling details to stdout,
    and writes a summary report.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run main.py first.")
        
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract profiles
    shape = df.shape
    columns = list(df.columns)
    missing_values = df.isnull().sum()
    data_types = df.dtypes
    summary_stats = df.describe(include='all')
    
    # Display to console
    print("\n" + "=" * 50)
    print("           DATASET PROFILE SUMMARY")
    print("=" * 50)
    print(f"Shape: {shape[0]} rows, {shape[1]} columns\n")
    print("Columns:")
    print(columns)
    print("\nData Types:")
    print(data_types.to_string())
    print("\nMissing Values:")
    print(missing_values.to_string())
    print("\nSummary Statistics:")
    print(summary_stats.to_string())
    print("=" * 50 + "\n")
    
    # Save output report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Data Profile Report\n\n")
        f.write("## Overview\n")
        f.write(f"Source file: `{csv_path}`\n")
        f.write(f"Dimensions: {shape[0]} rows, {shape[1]} columns\n\n")
        
        f.write("## Features and Data Types\n")
        f.write("| Feature | Data Type | Missing Values |\n")
        f.write("| --- | --- | --- |\n")
        for col in columns:
            f.write(f"| {col} | {data_types[col]} | {missing_values[col]} |\n")
        f.write("\n")
        
        f.write("## Summary Statistics\n")
        f.write("```\n")
        f.write(summary_stats.to_string())
        f.write("\n```\n")
        
    print(f"Data profile report saved successfully to {report_path}")
    return df

if __name__ == "__main__":
    load_and_profile_data()
