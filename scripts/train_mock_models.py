import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# Ensure the models directory exists
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)

print("Starting model training with synthetic data...")

# ==========================================
# 1. Login Risk Model (Synthetic Data)
# ==========================================
print("Generating synthetic data for Login Risk...")
# Features: user_id, ip_address, user_agent, device_type, country, rtt_ms
# To simplify the model, we encode categorical variables as simple numerical identifiers for this mock.
# In a real scenario, you would use OneHotEncoding, Hashing, or Target Encoding.

num_login_samples = 2000
np.random.seed(42)

# Generate features
login_X = pd.DataFrame({
    'user_id': np.random.randint(1, 1000, num_login_samples),
    # For IP, User-Agent, Country, Device, we'll hash/encode them to simple ints for the mock model
    'ip_hash': np.random.randint(1, 500, num_login_samples),
    'ua_hash': np.random.randint(1, 50, num_login_samples), 
    'device_type': np.random.randint(0, 3, num_login_samples), # 0: Desktop, 1: Mobile, 2: Tablet
    'country_code': np.random.randint(0, 10, num_login_samples), # 10 countries
    'rtt_ms': np.random.exponential(scale=100, size=num_login_samples).astype(int) # Response time
})

# Generate labels (0: Legitimate, 1: High Risk/MFA, 2: Block)
# We make it slightly realistic: high rtt or weird countries correlate with higher risk
login_y = np.where(login_X['rtt_ms'] > 400, 1, 0)
login_y = np.where(login_X['country_code'] > 7, login_y + 1, login_y)
login_y = np.clip(login_y, 0, 1) # Binary classification for simplicity: 0 or 1

print("Training Login Risk XGBoost Model...")
login_model = xgb.XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')
login_model.fit(login_X, login_y)

login_model_path = os.path.join(models_dir, 'login_risk_model.pkl')
joblib.dump(login_model, login_model_path)
print(f"Saved Login Risk Model to {login_model_path}")

# ==========================================
# 2. Transaction Fraud Model (Synthetic Data)
# ==========================================
print("\nGenerating synthetic data for Transaction Fraud...")
# Features: user_id, order_id, total_amount, payment_method, product_category, customer_age

num_tx_samples = 3000

tx_X = pd.DataFrame({
    'user_id': np.random.randint(1, 1000, num_tx_samples),
    'order_id': np.arange(1, num_tx_samples + 1),
    'total_amount': np.random.lognormal(mean=4, sigma=1, size=num_tx_samples),
    'payment_method': np.random.randint(0, 4, num_tx_samples), # 0: Card, 1: PayPal, 2: Crypto, 3: Wallet
    'product_category': np.random.randint(0, 5, num_tx_samples),
    'customer_age': np.random.randint(18, 70, num_tx_samples)
})

# Fraud is rare. We create an artificial rule: Very high amount + specific payment method = risky
tx_y = np.where((tx_X['total_amount'] > 500) & (tx_X['payment_method'] == 2), 1, 0) # Crypto high amounts
tx_y = np.where(tx_X['total_amount'] > 2000, 1, tx_y) # Extreme amounts are always risky
tx_X.drop('order_id', axis=1, inplace=True) # Order ID isn't a good feature for prediction

print("Training Transaction Fraud XGBoost Model...")
tx_model = xgb.XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')
tx_model.fit(tx_X, tx_y)

tx_model_path = os.path.join(models_dir, 'transaction_fraud_model.pkl')
joblib.dump(tx_model, tx_model_path)
print(f"Saved Transaction Fraud Model to {tx_model_path}")

print("\nDone! Ensure 'services/prediction_service.py' is updated to load these files.")
