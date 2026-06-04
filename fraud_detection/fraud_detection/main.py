import os
import sys

# Ensure the current directory is in the Python search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_synthetic_data, save_data
from src.feature_engineering import engineer_features
from src.preprocessing import preprocess_data
from src.train import train_and_compare_models
from src.evaluate import evaluate_model

def main():
    print("=" * 60)
    print("         STARTING FRAUD DETECTION PIPELINE")
    print("=" * 60)
    
    # Define file and directory paths
    raw_data_path = os.path.join("data", "raw", "transactions.csv")
    raw_engineered_path = os.path.join("data", "raw", "transactions_engineered.csv")
    processed_dir = os.path.join("data", "processed")
    models_dir = os.path.join("models")
    model_path = os.path.join(models_dir, "fraud_detector_model.joblib")
    visualizations_dir = os.path.join("visualizations")
    reports_dir = os.path.join("reports")
    
    # Step 1: Synthetic Data Generation
    print("\n--- STEP 1: Generating Synthetic Data ---")
    data = generate_synthetic_data(num_samples=15000, random_seed=42)
    save_data(data, raw_data_path)
    
    # Step 2: Feature Engineering
    print("\n--- STEP 2: Engineering Fraud-Related Features ---")
    engineer_features(raw_data_path, raw_engineered_path, os.path.join(reports_dir, "feature_engineering_report.md"))
    
    # Step 3: Preprocessing & Scaling
    print("\n--- STEP 3: Preprocessing Features ---")
    preprocess_data(raw_engineered_path, processed_dir, models_dir)
    
    # Step 4: Model Training & Selection
    print("\n--- STEP 4: Training & Comparing ML Models ---")
    train_and_compare_models(processed_dir, models_dir)
    
    # Step 5: Model Evaluation and Plot Generation
    print("\n--- STEP 5: Evaluating Model & Generating Artifacts ---")
    evaluate_model(model_path, processed_dir, visualizations_dir, reports_dir)
    
    print("\n" + "=" * 60)
    print("          PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Serialized Model:  {model_path}")
    print(f"Visualizations:    {visualizations_dir}/")
    print(f"Evaluation Report: {reports_dir}/evaluation_report.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
