import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=15000, random_seed=42):
    """
    Generates a synthetic dataset representing credit card transactions with fraud labels,
    associating transactions with unique users chronologically over a 30-day period.
    """
    np.random.seed(random_seed)
    
    # 1. Transaction IDs
    transaction_ids = [f"TX{i:06d}" for i in range(num_samples)]
    
    # 2. Users Pool (500 unique users)
    num_users = 500
    user_ids = [f"USR{i:03d}" for i in range(num_users)]
    user_pool = np.random.choice(user_ids, size=num_samples)
    
    # Map users to ages
    user_ages = {uid: int(np.clip(np.random.normal(loc=41, scale=12), 18, 90)) for uid in user_ids}
    age = [user_ages[uid] for uid in user_pool]
    
    # 3. Timestamps (over a 30-day span in May 2026)
    start_date = pd.Timestamp("2026-05-01")
    random_seconds = np.random.randint(0, 30 * 24 * 3600, size=num_samples)
    timestamps = [start_date + pd.Timedelta(seconds=int(s)) for s in random_seconds]
    
    # Create Base DataFrame
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'user_id': user_pool,
        'age': age,
        'timestamp': timestamps
    })
    
    # Sort chronologically per user
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    
    # Derive hour and day of week
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 4. Numerical variables
    amount = np.random.lognormal(mean=3.5, sigma=1.0, size=num_samples)
    df['amount'] = np.round(amount, 2)
    
    distance_from_home = np.random.lognormal(mean=1.5, sigma=1.2, size=num_samples)
    df['distance_from_home'] = np.round(distance_from_home, 2)
    
    # 5. Categorical variables
    categories = ['grocery', 'entertainment', 'online_retail', 'travel', 'gas', 'services']
    category_probs = [0.35, 0.15, 0.25, 0.05, 0.15, 0.05]
    df['category'] = np.random.choice(categories, size=num_samples, p=category_probs)
    
    device_types = ['pos', 'web', 'mobile']
    device_probs = [0.50, 0.20, 0.30]
    df['device'] = np.random.choice(device_types, size=num_samples, p=device_probs)
    
    # 6. Fraud risk score logic
    risk_score = np.zeros(num_samples)
    risk_score += (df['amount'] > 200) * 2.5
    risk_score += (df['amount'] > 800) * 4.0
    risk_score += (df['distance_from_home'] > 50) * 3.0
    risk_score += (df['distance_from_home'] > 200) * 5.0
    risk_score += ((df['hour'] >= 1) & (df['hour'] <= 5)) * 2.0
    risk_score += (df['category'] == 'travel') * 1.5
    risk_score += (df['category'] == 'online_retail') * 1.0
    
    # Base risk plus random noise
    risk_score += np.random.normal(loc=-6.5, scale=2.0, size=num_samples)
    
    # Map to probability and draw binary target labels
    prob_fraud = 1 / (1 + np.exp(-risk_score))
    df['is_fraud'] = np.random.binomial(1, prob_fraud)
    
    return df

def save_data(df, output_path):
    """Saves the dataframe to the specified CSV path, ensuring parent folders exist."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Fraud count: {df['is_fraud'].sum()} ({df['is_fraud'].mean() * 100:.2f}%)")

if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "transactions.csv")
    data = generate_synthetic_data(num_samples=15000)
    save_data(data, raw_data_path)
