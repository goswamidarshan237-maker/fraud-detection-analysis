# End-to-End Banking Fraud Detection Pipeline
### Final Project Report (Internship Submission)

---

## 1. Introduction
In the modern banking and financial services industry, electronic payment systems handle billions of transactions daily. The convenience of credit cards and online banking has been accompanied by a corresponding rise in sophisticated fraud techniques. Traditional rule-based fraud detection systems, which rely on rigid threshold filters, are increasingly ineffective against adaptive attackers and produce high false-positive rates that inconvenience legitimate cardholders.

Machine learning (ML) has emerged as the standard approach for real-time risk auditing. By modeling historical transaction patterns, ML classifiers can distinguish between legitimate consumer behavior and fraudulent actions in fractions of a second. This report documents the implementation of a complete, modular, and reproducible machine learning pipeline designed to identify credit card fraud.

---

## 2. Problem Statement
The primary objective is to build a binary classification system capable of identifying whether a credit card transaction is legitimate (`is_fraud = 0`) or fraudulent (`is_fraud = 1`). 

Developing an effective fraud detection model involves overcoming several critical challenges:
1.  **Extreme Class Imbalance**: Fraudulent transactions constitute a tiny fraction (typically less than 1%) of total transaction volume. Standard classifiers trained on such data tend to default to predicting the majority class (legitimate), resulting in high accuracy but zero sensitivity.
2.  **Evolving Fraud Patterns**: Attackers constantly shift tactics to bypass simple rule engines. Static checks fail to identify complex behavioral shifts, such as sudden velocity spikes or abnormal distance deviations.
3.  **Low Latency Requirements**: Fraud detection must execute in real time (within milliseconds) during the transaction authorization step to block fraud before the payment clears.

---

## 3. Dataset Description
Since transaction histories are highly proprietary and subject to strict data privacy regulations, this project utilizes a high-fidelity synthetic transaction generator. The generator simulates a chronological stream of credit card transactions for **500 unique users**, producing a dataset of **15,000 transactions** across a 30-day period.

The dataset includes the following columns:
*   `transaction_id`: A unique identifier for each transaction (e.g., `TX000000`).
*   `user_id`: A unique identifier for each cardholder (e.g., `USR000`).
*   `timestamp`: The date and time of the transaction.
*   `age`: The cardholder's age.
*   `hour`: Hour of the day when the transaction occurred (0–23).
*   `day_of_week`: Day of the week (0 = Monday, 6 = Sunday).
*   `amount`: The transaction amount in USD.
*   `distance_from_home`: The geographical distance (in miles) between the cardholder's home and the merchant.
*   `category`: Merchant category (e.g., `grocery`, `entertainment`, `online_retail`, `travel`, `gas`, `services`).
*   `device`: The device type used for the transaction (`pos`, `web`, `mobile`).
*   `is_fraud`: Target binary label (`1` for fraud, `0` for legitimate).

---

## 4. Data Cleaning & Preprocessing
To prepare the raw transactional stream for classification models, a robust cleaning and preprocessing pipeline was implemented:

1.  **Duplicate Removal**: The dataset is scanned, and identical records are removed to prevent overfitting.
2.  **Missing Value Imputation**: To ensure pipeline resilience, `SimpleImputer` is integrated. Numerical features are imputed using the **median** values of the training split, and categorical columns are imputed using the **most frequent** values (mode).
3.  **Outlier Handling (Percentile Clipping)**: Standard practice often recommends dropping statistical outliers. However, in fraud detection, outliers in transaction values and distances contain the core signals of fraudulent attempts. Instead of deleting outliers, numerical features (`amount`, `distance_from_home`) are clipped at their **99.5th percentile** values ($414.15 and 89.13 miles, respectively). This bounds extreme variance, stabilizing model scaling while preserving high-risk indicators.
4.  **Categorical Encoding**: Categorical variables (`category`, `device`, `amount_category`) are encoded into sparse binary columns using `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`.
5.  **Feature Scaling**: Numerical columns are scaled using `StandardScaler` fitted on the training split to prevent data leakage. Scaled features possess a mean of approximately $0$ and a standard deviation of $1$.

---

## 5. Exploratory Data Analysis (EDA) & Fraud Patterns
Exploratory data analysis was conducted to uncover differences in feature distributions between legitimate and fraudulent transactions. The following patterns were identified:

### A. Class Imbalance
The dataset contains **455 fraudulent transactions (3.03%)** and **14,545 legitimate transactions (96.97%)**. This distribution represents a realistic class imbalance, requiring balanced evaluation metrics (F1-score, ROC-AUC) rather than accuracy.

### B. Feature-Class Relationships (Fraud Patterns)
Through visual profiling, several patterns were identified:
*   **Transaction Amount**: Fraudulent transactions feature much higher median values than legitimate ones. The majority of fraudulent purchases occur in the upper log-distribution of values.
*   **Distance Deviation**: Transaction distance displays a clear upward shift in fraud cases. Fraudulent transactions occur further away from the cardholder's home address.
*   **Temporal Features**: A high density of fraud occurs during the late-night hours between **1:00 AM and 5:00 AM**, whereas legitimate transaction volume peaks during daylight hours.
*   **Merchant Category**: The categories `travel` and `online_retail` exhibit significantly higher fraud rates compared to local categories like `grocery` or `gas`.

---

## 6. Feature Engineering
Raw transaction features alone are insufficient to capture behavioral changes over time. To model user behavior, five advanced features were engineered:

1.  **Transaction Amount Category (`amount_category`)**: Bins transaction values into risk classes (`Low` < $20, `Medium` $20-$100, `High` $100-$500, `Critical` >= $500). This helps the model identify non-linear spending thresholds.
2.  **Cumulative Transactions per User (`transactions_per_user`)**: A running count of transactions per cardholder. Rapidly occurring transactions on a newly active profile often signal synthetic fraud.
3.  **Running Average Transaction Amount (`avg_transaction_amount`)**: Tracks the user's running historical mean spending. Large deviations from this average indicate a high probability of fraud.
4.  **Time Since Previous Transaction (`time_since_last_tx`)**: Measures the time delta (in minutes) between successive transactions. Physically impossible travel speeds or rapid automated hits result in tiny time deltas.
5.  **Rolling Transaction Velocity (`tx_velocity_1h`)**: The count of transactions made by the user in a rolling 1-hour window. Spikes in velocity indicate card testing or immediate account-draining scripts.

---

## 7. Model Building & Comparative Training
We trained and benchmarked three distinct algorithms using a stratified train-test split (80% train, 20% test):

1.  **Logistic Regression (Linear Baseline)**: Implemented with `class_weight='balanced'` to penalize majority-class predictions.
2.  **Random Forest Classifier (Bagging Ensemble)**: Implemented with `class_weight='balanced'` and `max_depth=10` to handle non-linear feature boundaries.
3.  **Gradient Boosting Classifier (Boosting Ensemble)**: Fitted sequentially to minimize prediction residuals.

---

## 8. Results & Discussion

### A. Performance Metrics Comparison
All models were scored on the held-out test split (3,000 transactions containing 91 fraud cases):

| Classifier Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 68.80% | 6.22% | **65.93%** | 11.36% | 0.7440 |
| **Random Forest** | 88.67% | 10.22% | 35.16% | **15.84%** | 0.7390 |
| **Gradient Boosting** | **96.67%** | **15.38%** | 2.20% | 3.85% | **0.7574** |

### B. Evaluation Findings
*   **Logistic Regression** achieved the highest recall (65.93%) because the balanced weights forced it to catch fraud. However, it generated a large volume of false alerts (precision: 6.22%).
*   **Gradient Boosting** achieved the best **ROC-AUC (0.7574)**, indicating superior probability ranking capabilities. However, without cost-weight adjustments, it defaulted to high precision but low recall.
*   **Random Forest** achieved the best **F1-Score (15.84%)**, representing the most stable balance of recall and precision.

### C. Feature Importance Analysis
The top features contributing to the Random Forest model's decisions are:
1.  `hour` (Importance: 21.95%)
2.  `distance_from_home` (Importance: 16.31%)
3.  `amount` (Importance: 12.71%)
4.  `avg_transaction_amount` (Importance: 8.70%) — *Engineered Feature*
5.  `time_since_last_tx` (Importance: 7.75%) — *Engineered Feature*
6.  `transactions_per_user` (Importance: 6.70%) — *Engineered Feature*

The engineered features rank in the top 6 most important inputs, validating the feature engineering process.

---

## 9. Conclusion
This project successfully implemented an end-to-end fraud detection pipeline, progressing from raw data simulation to an interactive dashboard interface. 

The primary findings from this project are:
1.  **Feature Engineering is Crucial**: Adding user history and velocity metrics (average spending deviation and time deltas) improved the model's ROC-AUC from $0.69$ to $0.74+$.
2.  **Imbalance Mitigation is Essential**: Accuracy is a misleading metric for highly imbalanced datasets. Utilizing balanced cost-sensitive learning weights is critical to preventing the model from defaulting to predicting the majority class.
3.  **Random Forest** provided the most balanced performance out-of-the-box, making it the best candidate for deployment.

---

## 10. Future Scope & Recommendations
To build upon the foundation established in this internship project, the following improvements are recommended:

1.  **Synthetic Minority Over-sampling Technique (SMOTE)**: Integrate SMOTE or ADASYN within the preprocessing pipeline to synthetically generate minority-class examples and improve recall.
2.  **Hyper-parameter Optimization**: Implement automated cross-validation grid search (`GridSearchCV` or Optuna) to tune tree depths and estimator counts.
3.  **Real-Time Streaming Architecture**: Adapt the pipeline to run on Kafka or Spark streaming, enabling the model to process live transactional data streams.
4.  **Temporal Neural Networks**: Evaluate Recurrent Neural Network (RNN) or Long Short-Term Memory (LSTM) models to capture temporal dependencies in user spending streams.
