# Fraud Detection Machine Learning Pipeline

This repository implements a modular, end-to-end Machine Learning pipeline to identify credit card fraud. Since real-world transaction logs are proprietary, the pipeline includes a synthetic data generator to simulate highly imbalanced transaction details (e.g. amount, distance from home, transaction hour, categories, etc.) with specific fraud patterns.

A Random Forest Classifier trained with balanced class weights is used to model the transactions, and various metric visualizations (ROC Curve, PR-AUC, Confusion Matrix, and Feature Importances) are generated dynamically.

---

## Project Structure

The project has been structured according to industry standards for machine learning repositories:

```text
d:/Projects/Data Analytics/
├── data/                       # Dataset directories
│   ├── raw/                    # Untouched generated transaction data
│   └── processed/              # Normalized, encoded training/test splits
├── models/                     # Saved models (joblib format) & preprocessing states
├── notebooks/                  # Interactive Jupyter Notebooks for exploration
│   └── exploratory_analysis.ipynb
├── reports/                    # Generated summary documents
│   └── evaluation_report.md
├── src/                        # Modular source code
│   ├── __init__.py             # Package declaration
│   ├── data_generator.py       # Simulates realistic card transaction records
│   ├── preprocessing.py        # Pipelines scaling and encoding (prevents data leakage)
│   ├── train.py                # Handles training and handles imbalanced data
│   └── evaluate.py             # Generates evaluation reports and visualization figures
├── visualizations/             # Plot PNG files (ROC curve, PR curve, confusion matrix)
├── README.md                   # Project documentation
├── requirements.txt            # Package dependencies
└── main.py                     # Entry point to execute the pipeline end-to-end
```

---

## Getting Started

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Setup Virtual Environment (Recommended)
Open a terminal (or PowerShell on Windows) in this folder and create a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## How to Run

To run the complete pipeline (Data Generation $\rightarrow$ Preprocessing $\rightarrow$ Training $\rightarrow$ Evaluation), execute the root-level script:

```bash
python main.py
```

### Modular Execution
Alternatively, you can run individual stages separately:

1. **Generate Data:**
   ```bash
   python src/data_generator.py
   ```
2. **Preprocess:**
   ```bash
   python src/preprocessing.py
   ```
3. **Train:**
   ```bash
   python src/train.py
   ```
4. **Evaluate:**
   ```bash
   python src/evaluate.py
   ```

---

## Pipeline Details

1. **`src/data_generator.py`**: Simulates transactions. Imbalances are modeled so that approximately 1.5% of transactions are fraud. Fraud flags are generated using a logistic score containing rules (e.g. higher transaction amounts, transactions far from home, unusual hours between 1 AM and 5 AM).
2. **`src/preprocessing.py`**: Encodes categorical variables (device type, merchant category) using `OneHotEncoder` and scales numerical columns (age, transaction hour, amount, distance) with `StandardScaler` fitted on the training split to avoid data leakage.
3. **`src/train.py`**: Trains a `RandomForestClassifier` with balanced class weights so that the classifier adjusts its cost function dynamically to the small class of fraud cases.
4. **`src/evaluate.py`**: Scores the classifier on the held-out test split, exports the curves to `visualizations/`, and logs a full report in `reports/evaluation_report.md`.

---

## Exploratory Data Analysis

A starter notebook is available in `notebooks/exploratory_analysis.ipynb`. You can launch it using:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```
or inside your IDE to visually explore the distributions and correlations of features.
