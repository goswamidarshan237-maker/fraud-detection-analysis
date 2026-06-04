import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Fraud Analytics & Prediction Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Load data
@st.cache_data
def load_raw_data():
    raw_path = "data/raw/transactions_engineered.csv"
    if not os.path.exists(raw_path):
        raw_path = "data/raw/transactions.csv"
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return None

df = load_raw_data()

# Load models and preprocessors
@st.cache_resource
def load_ml_artifacts():
    try:
        model = joblib.load("models/best_fraud_detector_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        encoder = joblib.load("models/encoder.joblib")
        return model, scaler, encoder
    except Exception as e:
        return None, None, None

model, scaler, encoder = load_ml_artifacts()

# Sidebar Navigation
st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio("Go to:", ["📊 Fraud Analytics Dashboard", "🔍 Real-Time Prediction Page"])

if page == "📊 Fraud Analytics Dashboard":
    st.title("📊 Credit Card Fraud Analytics Dashboard")
    st.markdown("This dashboard provides overview metrics and distributions of credit card transaction fraud.")
    
    if df is not None:
        # KPI calculations
        total_tx = len(df)
        fraud_df = df[df['is_fraud'] == 1]
        fraud_rate = (len(fraud_df) / total_tx) * 100
        total_fraud_amount = fraud_df['amount'].sum()
        avg_tx_amount = df['amount'].mean()
        
        # 1. KPI Metric Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total_tx:,}")
        col2.metric("Overall Fraud Rate", f"{fraud_rate:.2f}%")
        col3.metric("Total Fraud Amount", f"${total_fraud_amount:,.2f}", delta_color="inverse")
        col4.metric("Avg Transaction Value", f"${avg_tx_amount:.2f}")
        
        st.write("---")
        
        # 2. Charts Row
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Hourly Transaction Density & Fraud Occurrences")
            hourly_counts = df.groupby('hour')['is_fraud'].agg(['count', 'sum'])
            hourly_counts.columns = ['Total Transactions', 'Fraud Transactions']
            st.bar_chart(hourly_counts)
            
        with col_right:
            st.subheader("Fraud Rate (%) by Transaction Category")
            cat_stats = df.groupby('category')['is_fraud'].mean() * 100
            st.bar_chart(cat_stats)
            
        st.write("---")
        
        # 3. Visualizations folder plots
        st.subheader("Model Evaluation Curve Benchmarks")
        plot_type = st.selectbox("Select Benchmark Plot to View:", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance"])
        
        plot_files = {
            "Confusion Matrix": "visualizations/confusion_matrix.png",
            "ROC Curve": "visualizations/roc_curve.png",
            "Precision-Recall Curve": "visualizations/precision_recall_curve.png",
            "Feature Importance": "visualizations/feature_importance.png"
        }
        
        selected_file = plot_files[plot_type]
        if os.path.exists(selected_file):
            st.image(selected_file, caption=f"{plot_type} Plot", use_container_width=True)
        else:
            st.warning(f"Plot file not found at {selected_file}. Run main.py first.")
            
    else:
        st.error("Transaction database not found! Please run the pipeline script (main.py) to generate synthetic transaction logs first.")

elif page == "🔍 Real-Time Prediction Page":
    st.title("🔍 Real-Time Fraud Prediction Tool")
    st.markdown("Use this interface to evaluate individual card transactions against our trained classification model.")
    
    if model is None or scaler is None or encoder is None:
        st.error("Machine learning model or preprocessing artifacts are missing! Please execute the pipeline training stage first.")
    else:
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            st.subheader("Transaction Profile")
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=10000.0, value=45.0, step=1.0)
            distance = st.number_input("Distance from Cardholder Home (miles)", min_value=0.1, max_value=2000.0, value=12.5, step=1.0)
            category = st.selectbox("Merchant Category", ['grocery', 'entertainment', 'online_retail', 'travel', 'gas', 'services'])
            device = st.selectbox("Transaction Device", ['pos', 'web', 'mobile'])
            hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
            day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            day_code = day_map[day_of_week]
            
        with col_in2:
            st.subheader("Cardholder Profile & Stream Velocity")
            age = st.slider("Cardholder Age", 18, 90, 35)
            tx_per_user = st.number_input("Cumulative Historical Transactions", min_value=1, max_value=10000, value=25)
            avg_tx_amount = st.number_input("Average Historical Spending ($)", min_value=1.0, max_value=10000.0, value=50.0)
            time_since_last = st.number_input("Minutes Since Last Transaction", min_value=-1.0, max_value=43200.0, value=120.0)
            velocity_1h = st.number_input("Transactions in Last 1 Hour", min_value=0, max_value=100, value=0)
            
        # Inference Logic
        if st.button("🔍 Evaluate Transaction"):
            # 1. Derive amount_category
            if amount < 20:
                amount_cat = 'Low'
            elif amount < 100:
                amount_cat = 'Medium'
            elif amount < 500:
                amount_cat = 'High'
            else:
                amount_cat = 'Critical'
                
            # 2. Assemble DataFrame
            input_df = pd.DataFrame([{
                'age': age,
                'hour': hour,
                'day_of_week': day_code,
                'amount': amount,
                'distance_from_home': distance,
                'transactions_per_user': tx_per_user,
                'avg_transaction_amount': avg_tx_amount,
                'time_since_last_tx': time_since_last,
                'tx_velocity_1h': velocity_1h,
                'category': category,
                'device': device,
                'amount_category': amount_cat
            }])
            
            # Define columns
            num_cols = ['age', 'hour', 'day_of_week', 'amount', 'distance_from_home', 'transactions_per_user', 'avg_transaction_amount', 'time_since_last_tx', 'tx_velocity_1h']
            cat_cols = ['category', 'device', 'amount_category']
            
            # Preprocess numericals and categoricals
            try:
                # Scale
                scaled_num = scaler.transform(input_df[num_cols])
                # Encode
                encoded_cat = encoder.transform(input_df[cat_cols])
                # Stack
                processed_row = np.hstack([scaled_num, encoded_cat])
                
                # Predict
                prob = model.predict_proba(processed_row)[0, 1]
                pred = 1 if prob >= 0.5 else 0
                
                st.write("---")
                st.subheader("Model Decision & Assessment")
                
                # High-end styling UI cards
                if pred == 1:
                    st.error(f"🚨 **FRAUD DETECTED (High Risk)** \n\n*   **Risk Probability:** {prob*100:.2f}%\n*   **Status:** Transaction blocked.")
                else:
                    st.success(f"✅ **TRANSACTION APPROVED (Low Risk)** \n\n*   **Risk Probability:** {prob*100:.2f}%\n*   **Status:** Transaction processed successfully.")
            except Exception as e:
                st.error(f"Inference pipeline execution error: {e}")
