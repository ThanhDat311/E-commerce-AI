import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import hashlib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Create a stable hashing mechanism for categorical labels exactly the same as test inference
def stable_hash(val):
    if pd.isna(val) or val == '' or val is None:
        return 0
    # MD5 hash string to a stable integer
    return int(hashlib.md5(str(val).encode('utf-8')).hexdigest(), 16) % (10**7)

def get_column_by_keyword(columns, keyword):
    for col in columns:
        if keyword.lower() in col.lower():
            return col
    return None

def preprocess_and_train():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'balanced_rba.csv')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    start_time = time.time()
    
    print(f"Loading balanced dataset from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print(f"Dataset Loaded. Shape: {df.shape}. Preprocessing data...")
    
    # 1. Identity Columns
    user_id_col = get_column_by_keyword(df.columns, 'User ID') or 'User ID'
    ip_col = get_column_by_keyword(df.columns, 'IP Address') or 'IP Address'
    ua_col = get_column_by_keyword(df.columns, 'User Agen') or 'User Agent String'
    device_col = get_column_by_keyword(df.columns, 'Device Type') or 'Device Type'
    country_col = get_column_by_keyword(df.columns, 'Country') or 'Country'
    rtt_col = get_column_by_keyword(df.columns, 'Round-Trip') or 'Round-Trip Time [ms]'
    target_col = get_column_by_keyword(df.columns, 'Account Takeover') or 'Is Account Takeover'

    features = pd.DataFrame()
    
    print("Mapping strings to stable integer hashes...")
    features['user_id_hash'] = df[user_id_col].apply(stable_hash)
    features['ip_hash'] = df[ip_col].apply(stable_hash)
    features['ua_hash'] = df[ua_col].apply(stable_hash)
    
    # Device mapping: 0=desktop, 1=mobile, 2=other
    features['device_type'] = df[device_col].apply(lambda x: 0 if 'desktop' in str(x).lower() else (1 if 'mobile' in str(x).lower() else 2))
    
    features['country_hash'] = df[country_col].apply(stable_hash)
    
    # Parse RTT: some values might be dashes '-' or NaNs, coerce them to numbers 
    if rtt_col in df.columns:
        features['rtt_ms'] = pd.to_numeric(df[rtt_col], errors='coerce').fillna(50)
    else:
        features['rtt_ms'] = 50

    # Extract Target
    # Labels in RBA usually use TRUE / FALSE strings or booleans
    y = df[target_col].astype(str).str.upper() == 'TRUE'
    y = y.astype(int)
    
    # XGBoost scales weights to handle imbalance where ATO (class 1) is rare
    print(f"Target variable extraction complete. Total Account Takeovers in chunk: {y.sum()}")
    
    # Split data to evaluate model
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=y)
    
    # ─── APPLY SMOTE FOR EXTREME IMBALANCE ─────────────────────────────────────
    print(f"\n--- Applying SMOTE ---")
    print(f"Before SMOTE - Train Class Distribution:\n{y_train.value_counts()}")
    
    # SMOTE's KNN algorithm requires normalized (scaled) data to compute distances accurately.
    # We fit the scaler strictly on the training set to prevent data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to the scaled training data.
    # sampling_strategy='minority' means we generate synthetic samples for the minority class (ATO)
    # until it has the same number of samples as the majority class.
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"After SMOTE - Train Class Distribution:\n{y_train_smote.value_counts()}")
    # ───────────────────────────────────────────────────────────────────────────
    
    print(f"\nTraining Login Risk XGBoost Model on {len(X_train_smote)} balanced rows...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        # scale_pos_weight is NO LONGER NEEDED because SMOTE perfectly balanced the classes
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train_smote, y_train_smote)
    
    print("\n--- Model Evaluation (Test Set: {} rows) ---".format(len(X_test_scaled)))
    
    # Note: We must predict using the identically scaled test set
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy Score : {acc * 100:.2f}%")
    print(f"F1-Score (Fraud): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("---------------------------------------------\n")
    
    # Create an inference pipeline so prediction code in Laravel/FastAPI doesn't have to change.
    # When predict() is called, this pipeline will automatically Scale the new data, then run XGBoost.
    from sklearn.pipeline import Pipeline
    inference_pipeline = Pipeline([
        ('scaler', scaler),
        ('xgb_model', model)
    ])
    
    model_save_path = os.path.join(models_dir, 'login_risk_model.pkl')
    joblib.dump(inference_pipeline, model_save_path)
    print(f"Real RBA Model saved successfully to {model_save_path}")
    print(f"Total processing time: {round(time.time() - start_time, 2)} seconds")

if __name__ == "__main__":
    preprocess_and_train()
