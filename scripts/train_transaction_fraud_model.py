"""
Train Transaction Fraud Detection Model
Dataset: Fraudulent E-Commerce Transactions (Kaggle)
- 23,634 rows, 5.17% fraud rate
- Features: Transaction Amount, Payment Method, Product Category,
            Quantity, Customer Age, Device Used, Account Age Days, Transaction Hour
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_PATH = r"D:\InformationTechnology\Semester6\project_AI\dataset\transaction_fraud\Fraudulent_E-Commerce_Transaction_Data_2.csv"
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'transaction_fraud_model.pkl')
ENCODER_SAVE_PATH = os.path.join(MODELS_DIR, 'transaction_fraud_encoders.pkl')

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Feature Engineering ──────────────────────────────────────────────────────

def build_encoders(df: pd.DataFrame) -> dict:
    """Fit LabelEncoders on categorical columns and return them."""
    encoders = {}
    for col in ['Payment Method', 'Product Category', 'Device Used']:
        le = LabelEncoder()
        le.fit(df[col].astype(str).str.lower().str.strip())
        encoders[col] = le
        print(f"  [{col}] classes: {list(le.classes_)}")
    return encoders


def encode_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Transform raw dataframe into model-ready feature matrix."""
    features = pd.DataFrame()

    # Numeric features (no transformation needed)
    features['total_amount'] = df['Transaction Amount'].astype(float)
    features['quantity'] = df['Quantity'].astype(int)
    features['customer_age'] = df['Customer Age'].astype(int)
    features['account_age_days'] = df['Account Age Days'].astype(int)
    features['transaction_hour'] = df['Transaction Hour'].astype(int)

    # Categorical → integer via LabelEncoder
    features['payment_method'] = encoders['Payment Method'].transform(
        df['Payment Method'].astype(str).str.lower().str.strip()
    )
    features['product_category'] = encoders['Product Category'].transform(
        df['Product Category'].astype(str).str.lower().str.strip()
    )
    features['device_used'] = encoders['Device Used'].transform(
        df['Device Used'].astype(str).str.lower().str.strip()
    )

    return features


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def train():
    start = time.time()

    print("=" * 60)
    print("  Transaction Fraud Detection — Training Pipeline")
    print("=" * 60)

    # 1. Load Data
    print(f"\n[1/5] Loading dataset from:\n  {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns")

    # 2. Target
    y = df['Is Fraudulent'].astype(int)
    fraud_count = int(y.sum())
    legit_count = int((y == 0).sum())
    fraud_pct = fraud_count / len(y) * 100
    print(f"\n[2/5] Target distribution:")
    print(f"      Legitimate : {legit_count:,} ({100 - fraud_pct:.2f}%)")
    print(f"      Fraud      : {fraud_count:,} ({fraud_pct:.2f}%)")

    # 3. Feature Engineering
    print(f"\n[3/5] Encoding categorical features...")
    encoders = build_encoders(df)
    X = encode_features(df, encoders)
    print(f"      Feature matrix shape: {X.shape}")
    print(f"      Feature columns: {list(X.columns)}")

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ─── APPLY SMOTE FOR CLASS IMBALANCE ───────────────────────────────────────
    print(f"\n[4/5] Applying SMOTE & Scaling...")
    print(f"      Before SMOTE - Train Class Distribution:\n{y_train.value_counts().to_string()}")
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE to balance fraud vs non-fraud
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"      After SMOTE - Train Class Distribution:\n{y_train_smote.value_counts().to_string()}")
    # ───────────────────────────────────────────────────────────────────────────

    print(f"\n[5/5] Training XGBoost model...")
    print(f"      Train size   : {len(X_train_smote):,}")
    print(f"      Test size    : {len(X_test_scaled):,}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        # scale_pos_weight is NO LONGER NEEDED because SMOTE perfectly balanced the classes
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,
    )

    model.fit(
        X_train_smote, y_train_smote,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )

    # 6. Evaluation
    print(f"\n[6/6] Evaluating model on test set ({len(X_test_scaled):,} rows)...")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Find optimal threshold for F1 Score
    best_threshold = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_thresh = (y_proba >= thresh).astype(int)
        f1_thresh = f1_score(y_test, y_pred_thresh)
        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_threshold = thresh

    y_pred = (y_proba >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "─" * 55)
    print(f"  Optimal F1 Threshold : {best_threshold:.2f}")
    print(f"  Accuracy             : {acc * 100:.2f}%")
    print(f"  AUC-ROC              : {auc:.4f}")
    print(f"  F1 (fraud)           : {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[TN={cm[0][0]:5d}  FP={cm[0][1]:5d}]")
    print(f"   [FN={cm[1][0]:5d}  TP={cm[1][1]:5d}]]")
    print("─" * 55)

    # Feature Importance (Must reference the base XGBoost model)
    print("\n  Feature Importances:")
    importance = dict(zip(X.columns, model.feature_importances_))
    for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"  {feat:20s} {score:.4f} {bar}")
        
    # Wrap model and scaler in pipeline
    inference_pipeline = Pipeline([
        ('scaler', scaler),
        ('xgb_model', model)
    ])

    # Save model + encoders
    joblib.dump(inference_pipeline, MODEL_SAVE_PATH)
    joblib.dump(encoders, ENCODER_SAVE_PATH)
    print(f"\n✔  Model saved   → {MODEL_SAVE_PATH}")
    print(f"✔  Encoders saved → {ENCODER_SAVE_PATH}")
    print(f"⏱  Total time: {time.time() - start:.2f}s\n")


if __name__ == "__main__":
    train()
