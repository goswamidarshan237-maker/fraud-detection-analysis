import os
import pandas as pd
import numpy as np

def engineer_features(input_path="data/raw/transactions.csv", output_path="data/raw/transactions_engineered.csv", report_path="reports/feature_engineering_report.md"):
    """
    Creates fraud-specific engineered features from transaction streams.
    Includes amount categorization, cumulative historical counts, rolling averages,
    time deltas, and transaction velocity.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input raw dataset not found at {input_path}. Run data_generator.py first.")
        
    print(f"Loading raw data for feature engineering from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Ensure chronological order per user for sequence features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    
    # 1. Transaction Amount Category
    # Define bins: Low (<20), Medium (20-100), High (100-500), Critical (>=500)
    bins = [0, 20, 100, 500, np.inf]
    labels = ['Low', 'Medium', 'High', 'Critical']
    df['amount_category'] = pd.cut(df['amount'], bins=bins, labels=labels)
    
    # Grouped operations per user
    user_grouped = df.groupby('user_id')
    
    # 2. Transactions per User (Cumulative transaction count)
    df['transactions_per_user'] = user_grouped.cumcount() + 1
    
    # 3. Average Transaction Amount (Cumulative historical average)
    df['avg_transaction_amount'] = user_grouped['amount'].transform(lambda x: x.expanding().mean())
    df['avg_transaction_amount'] = np.round(df['avg_transaction_amount'], 2)
    
    # 4. Time Since Previous Transaction (in minutes)
    df['time_since_last_tx'] = user_grouped['timestamp'].diff().dt.total_seconds() / 60.0
    df['time_since_last_tx'] = df['time_since_last_tx'].fillna(-1.0) # Fill first transaction with -1
    df['time_since_last_tx'] = np.round(df['time_since_last_tx'], 2)
    
    # 5. Transaction Velocity (Number of transactions in a rolling 1-hour window)
    # To handle duplicate timestamps and correct mapping, we preserve original indices.
    df['orig_index'] = df.index
    df_sorted = df.sort_values(by=['user_id', 'timestamp'])
    
    def get_rolling_count(group):
        temp = group.set_index('timestamp')
        roll = temp['orig_index'].rolling('1h', closed='left').count()
        temp['tx_velocity_1h'] = roll
        return temp.reset_index()
        
    df_res = df_sorted.groupby('user_id', group_keys=False).apply(get_rolling_count)
    df_res = df_res.set_index('orig_index').sort_index()
    df['tx_velocity_1h'] = df_res['tx_velocity_1h'].fillna(0).astype(int)
    df = df.drop(columns=['orig_index'])
    
    # Save the engineered dataframe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Engineered data successfully saved to: {output_path}")
    
    # Save Feature Engineering Report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Feature Engineering Report\n\n")
        f.write("This report documents the engineered features designed to improve model sensitivity to credit card fraud, along with the domain-specific rationale for each feature.\n\n")
        
        f.write("## 1. Engineered Features & Descriptions\n\n")
        
        f.write("### A. Transaction Amount Category (`amount_category`)\n")
        f.write("- **Definition:** Categorizes transaction amounts into risk bins (`Low` < $20, `Medium` $20-$100, `High` $100-$500, `Critical` >= $500).\n")
        f.write("- **Why it helps:** Fraud transactions typically feature abnormally high prices (unusual purchases of high-value commodities). Risk binning allows classifiers to learn non-linear thresholds on value spikes directly.\n\n")
        
        f.write("### B. Cumulative Transactions per User (`transactions_per_user`)\n")
        f.write("- **Definition:** A running count of the total number of transactions recorded for each user.\n")
        f.write("- **Why it helps:** Established accounts (high transaction counts) show stable behavior patterns. A new card/account running multiple heavy transactions quickly (low cumulative counts but high frequency) is a classic indicator of synthetic profile fraud.\n\n")
        
        f.write("### C. Average Transaction Amount (`avg_transaction_amount`)\n")
        f.write("- **Definition:** The running average amount of all historical transactions made by the user.\n")
        f.write("- **Why it helps:** Fraudsters usually spend way more than a victim's normal transactional footprint. Comparing the current amount to the user's running historical average flags standard deviation spikes.\n\n")
        
        f.write("### D. Time Since Previous Transaction (`time_since_last_tx`)\n")
        f.write("- **Definition:** Time delta (in minutes) between the current transaction and the user's immediately preceding transaction. First transactions default to `-1`.\n")
        f.write("- **Why it helps:** Physically impossible velocity (e.g. card run in two locations within 2 minutes) or rapid repeated card testing (multiple attempts within seconds) yields tiny time intervals. A low delta is highly correlated with script-driven automated fraud.\n\n")
        
        f.write("### E. Rolling Transaction Velocity (`tx_velocity_1h`)\n")
        f.write("- **Definition:** Rolling count of transactions made by the user within the preceding 1 hour.\n")
        f.write("- **Why it helps:** Legitimate cardholders rarely execute multiple transactions in an hour. Multiple hits in a short window imply card testing or rapid asset draining.\n\n")
        
        f.write("## 2. Sample Data with Engineered Features\n")
        f.write("```\n")
        sample_cols = ['user_id', 'amount', 'amount_category', 'transactions_per_user', 'avg_transaction_amount', 'time_since_last_tx', 'tx_velocity_1h']
        f.write(df[sample_cols].head(15).to_string(index=False))
        f.write("\n```\n")
        
    print(f"Feature engineering report saved to {report_path}")
    return df

if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "transactions.csv")
    engineered_data_path = os.path.join("data", "raw", "transactions_engineered.csv")
    report_output_path = os.path.join("reports", "feature_engineering_report.md")
    
    if os.path.exists(raw_data_path):
        engineer_features(raw_data_path, engineered_data_path, report_output_path)
    else:
        print("Raw transaction data file not found.")
