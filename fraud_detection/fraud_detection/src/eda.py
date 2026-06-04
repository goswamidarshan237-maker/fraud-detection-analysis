import os
import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_path="data/raw/transactions.csv", output_dir="visualizations"):
    """
    Performs EDA on the raw dataset and exports key visualizations for reports.
    Generates class distribution (pie and bar charts), transaction amount (histogram),
    correlation (heatmap), numerical attributes (boxplots), and time trends.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run main.py first.")
        
    print(f"Running exploratory data analysis on {data_path}...")
    df = pd.read_csv(data_path)
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    colors = ['#4CAF50', '#F44336'] # Green for Legit (0), Red for Fraud (1)
    
    # 1. Pie Chart: Class Distribution
    plt.figure(figsize=(6, 6))
    counts = df['is_fraud'].value_counts()
    labels = ['Legitimate', 'Fraudulent']
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0, 0.1))
    plt.title('Distribution of Transaction Classes (Pie Chart)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_class_distribution_pie.png"), dpi=150)
    plt.close()
    
    # 2. Bar Chart: Class Counts
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=labels, y=counts.values, palette=colors, hue=labels, legend=False)
    plt.title('Distribution of Transaction Classes (Bar Chart)')
    plt.ylabel('Count')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_class_distribution_bar.png"), dpi=150)
    plt.close()
    
    # 3. Histogram: Amount Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='amount', hue='is_fraud', kde=True, bins=50, palette={0: '#4CAF50', 1: '#F44336'}, log_scale=True)
    plt.title('Transaction Amount Distribution (Log Scale)')
    plt.xlabel('Amount (Log Scale)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_amount_distribution_histogram.png"), dpi=150)
    plt.close()
    
    # 4. Correlation Heatmap
    df_corr = df.copy()
    # Code categorical columns as numerical integers for correlation analysis
    df_corr['category_code'] = df_corr['category'].astype('category').cat.codes
    df_corr['device_code'] = df_corr['device'].astype('category').cat.codes
    numerical_cols = ['age', 'hour', 'amount', 'distance_from_home', 'category_code', 'device_code', 'is_fraud']
    corr_matrix = df_corr[numerical_cols].corr()
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1.0, vmax=1.0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=150)
    plt.close()
    
    # 5. Daily Transaction Trends
    # Simulate dates over 30 days to plot transaction daily trends
    df_trends = df.copy()
    np.random.seed(42)
    dates = pd.date_range(start="2026-05-01", periods=30)
    df_trends['date'] = np.random.choice(dates, size=len(df_trends))
    daily_counts = df_trends.groupby('date').size()
    
    plt.figure(figsize=(10, 5))
    daily_counts.plot(kind='line', marker='o', color='#2196F3', linewidth=2)
    plt.title('Daily Transaction Volume Trend (30-Day Period)')
    plt.xlabel('Date')
    plt.ylabel('Number of Transactions')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_daily_trends.png"), dpi=150)
    plt.close()
    
    # 6. Hourly Fraud Trends
    hourly_stats = df.groupby('hour')['is_fraud'].agg(['count', 'sum'])
    hourly_stats.columns = ['total_transactions', 'fraud_transactions']
    hourly_stats['fraud_rate'] = (hourly_stats['fraud_transactions'] / hourly_stats['total_transactions']) * 100
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Total transaction bars
    color_bar = '#B0BEC5'
    ax1.bar(hourly_stats.index, hourly_stats['total_transactions'], color=color_bar, alpha=0.7, label='Total Transactions')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Total Transactions', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(range(24))
    
    # Fraud rate line overlay
    ax2 = ax1.twinx()
    color_line = '#E53935'
    ax2.plot(hourly_stats.index, hourly_stats['fraud_rate'], color=color_line, marker='x', linewidth=2, label='Fraud Rate (%)')
    ax2.set_ylabel('Fraud Rate (%)', color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)
    
    plt.title('Hourly Transaction Volume & Fraud Rate')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_hourly_trends.png"), dpi=150)
    plt.close()
    
    # 7. Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.boxplot(ax=axes[0], data=df, x='is_fraud', y='age', hue='is_fraud', palette=colors, legend=False)
    axes[0].set_title('Age by Class')
    axes[0].set_xlabel('Is Fraud')
    axes[0].set_ylabel('Age')
    
    sns.boxplot(ax=axes[1], data=df, x='is_fraud', y='amount', hue='is_fraud', palette=colors, legend=False)
    axes[1].set_title('Amount by Class (Log Scale)')
    axes[1].set_xlabel('Is Fraud')
    axes[1].set_ylabel('Amount')
    axes[1].set_yscale('log')
    
    sns.boxplot(ax=axes[2], data=df, x='is_fraud', y='distance_from_home', hue='is_fraud', palette=colors, legend=False)
    axes[2].set_title('Distance by Class (Log Scale)')
    axes[2].set_xlabel('Is Fraud')
    axes[2].set_ylabel('Distance')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_feature_boxplots.png"), dpi=150)
    plt.close()
    
    print(f"EDA successfully completed. Graphs saved to {output_dir}")

if __name__ == "__main__":
    run_eda()
