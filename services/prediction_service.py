from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import hashlib
import os
from core.config import settings

def stable_hash(val):
    if pd.isna(val) or val == '' or val is None:
        return 0
    return int(hashlib.md5(str(val).encode('utf-8')).hexdigest(), 16) % (10**7)

class LoginPredictionRequest(BaseModel):
    user_id: int
    ip_address: str
    user_agent: str
    device_type: str
    country: str | None = None
    rtt_ms: int | None = None

class TransactionPredictionRequest(BaseModel):
    user_id: int
    order_id: int
    total_amount: float
    payment_method: str
    product_category: Optional[str] = None
    customer_age: Optional[int] = None
    quantity: Optional[int] = 1
    account_age_days: Optional[int] = 365
    transaction_hour: Optional[int] = 12
    device_used: Optional[str] = None

class PredictionService:
    def __init__(self):
        self.login_model = None
        self.transaction_model = None
        self.transaction_encoders = None

        try:
            self.login_model = joblib.load(settings.LOGIN_RISK_MODEL_PATH)
            print("Login Risk Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load Login Risk Model: {e}")

        try:
            self.transaction_model = joblib.load(settings.TRANSACTION_FRAUD_MODEL_PATH)
            print("Transaction Fraud Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load Transaction Fraud Model: {e}")

        # Load encoders saved alongside the transaction fraud model
        encoder_path = os.path.join(
            os.path.dirname(settings.TRANSACTION_FRAUD_MODEL_PATH),
            'transaction_fraud_encoders.pkl'
        )
        try:
            self.transaction_encoders = joblib.load(encoder_path)
            print("Transaction Fraud Encoders loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load Transaction Encoders (fallback mapping will be used): {e}")
            self.transaction_encoders = None

    def predict_login_risk(self, data: LoginPredictionRequest) -> dict:
        """
        Evaluate risk of a login attempt.
        """
        if not self.login_model:
            return {"risk_score": 0.0, "auth_decision": "passive_auth_allow", "reasons": ["AI Model Offline (Fallback)"]}
            
        # Map Request model to features
        dev_map = {'desktop': 0, 'mobile': 1, 'tablet': 2}
        
        # Prepare DataFrame shape expected by XGBoost model
        features = pd.DataFrame([{
            'user_id_hash': stable_hash(data.user_id),
            'ip_hash': stable_hash(data.ip_address),
            'ua_hash': stable_hash(data.user_agent),
            'device_type': dev_map.get(data.device_type.lower(), 0),
            'country_hash': stable_hash(data.country or 'US'),
            'rtt_ms': data.rtt_ms or 50
        }])
        
        # Predict probability of class 1 (High Risk)
        prob = float(self.login_model.predict_proba(features)[0][1])
        risk_score = prob
        
        auth_decision = "passive_auth_allow"
        reasons = ["Automated AI Assessment"]
        
        if risk_score > 0.8:
            auth_decision = "block_access"
            reasons.append("High risk anomaly detected (potentially high RTT or unusual region)")
        elif risk_score > 0.6:
            auth_decision = "challenge_otp"
            reasons.append("Moderate risk detected")

        return {
            "risk_score": round(risk_score, 4),
            "auth_decision": auth_decision,
            "reasons": reasons
        }

    def predict_transaction_fraud(self, data: TransactionPredictionRequest) -> dict:
        """
        Evaluate fraud risk of a transaction using the real Kaggle-trained model.
        Features: total_amount, quantity, customer_age, account_age_days,
                  transaction_hour, payment_method, product_category, device_used
        """
        if not self.transaction_model:
            return {"risk_score": 0.0, "decision": "allow", "reasons": ["AI Model Offline (Fallback)"]}

        # ── Fallback numeric maps in case encoders are not loaded ──────────
        PAY_MAP = {
            'debit card': 0, 'credit card': 1,
            'paypal': 2, 'bank transfer': 3,
            # Legacy aliases kept for backward compatibility with Laravel
            'card': 1, 'crypto': 1, 'wallet': 3
        }
        CAT_MAP = {
            'clothing': 0, 'electronics': 1,
            'health & beauty': 2, 'home & garden': 3, 'toys & games': 4,
            'misc': 1
        }
        DEV_MAP = {'desktop': 0, 'mobile': 1, 'tablet': 2}

        pay_str = (data.payment_method or 'credit card').lower().strip()
        cat_str = (data.product_category or 'electronics').lower().strip()
        dev_str = (data.device_used or 'desktop').lower().strip()

        if self.transaction_encoders:
            enc = self.transaction_encoders
            def safe_encode(encoder, val, fallback_map):
                try:
                    return int(encoder.transform([val])[0])
                except ValueError:
                    return fallback_map.get(val, 0)
            pay_enc = safe_encode(enc['Payment Method'], pay_str, PAY_MAP)
            cat_enc = safe_encode(enc['Product Category'], cat_str, CAT_MAP)
            dev_enc = safe_encode(enc['Device Used'], dev_str, DEV_MAP)
        else:
            pay_enc = PAY_MAP.get(pay_str, 1)
            cat_enc = CAT_MAP.get(cat_str, 1)
            dev_enc = DEV_MAP.get(dev_str, 0)

        features = pd.DataFrame([{
            'total_amount':     data.total_amount,
            'quantity':         data.quantity if data.quantity is not None else 1,
            'customer_age':     data.customer_age if data.customer_age is not None else 30,
            'account_age_days': data.account_age_days if data.account_age_days is not None else 365,
            'transaction_hour': data.transaction_hour if data.transaction_hour is not None else 12,
            'payment_method':   pay_enc,
            'product_category': cat_enc,
            'device_used':      dev_enc,
        }])

        prob = float(self.transaction_model.predict_proba(features)[0][1])
        risk_score = prob

        decision = "allow"
        reasons = ["Automated AI Assessment"]

        if risk_score >= 0.60:
            decision = "block"
            reasons.append("High fraud probability — transaction flagged by AI model.")
        elif risk_score >= 0.35:
            decision = "review"
            reasons.append("Moderate fraud indicators detected — manual review recommended.")

        return {
            "risk_score": round(risk_score, 4),
            "decision": decision,
            "reasons": reasons
        }
