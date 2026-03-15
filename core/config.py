from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "E-commerce Fraud Detection AI"
    API_V1_STR: str = "/api/v1"
    API_KEY: str = "secret-ecommerce-ai-key-2026"
    
    # Model configs
    LOGIN_RISK_MODEL_PATH: str = "models/login_risk_model.pkl"
    TRANSACTION_FRAUD_MODEL_PATH: str = "models/transaction_fraud_model.pkl"

    class Config:
        case_sensitive = True

settings = Settings()
