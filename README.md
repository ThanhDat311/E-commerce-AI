# E-commerce Fraud Detection AI

A high-performance microservice providing AI-driven Risk Assessment and Fraud Detection for E-commerce platforms. Built with **FastAPI** and **XGBoost**.

## 🚀 Key Features

### 1. Login Risk Assessment (RBA)
- Evaluates the risk of login attempts based on user behavior and metadata.
- Features analyzed: User ID, IP Address, User Agent, Device Type, Country, and RTT (Round Trip Time).
- Decisions: `passive_auth_allow`, `challenge_otp`, or `block_access`.

### 2. Transaction Fraud Detection
- Detects fraudulent transactions in real-time.
- Features analyzed: Total Amount, Quantity, Customer Age, Account Age, Transaction Hour, Payment Method, Product Category, and Device Used.
- Trained on Kaggle dataset using **SMOTE** for class imbalance handling and **XGBoost** for high accuracy.
- Decisions: `allow`, `review`, or `block`.

## 🛠 Tech Stack
- **Framework**: FastAPI (Asynchronous API)
- **Machine Learning**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Security**: API Key authentication
- **Serialization**: Joblib

## 📂 Project Structure
- `api/`: REST API routing and security.
- `core/`: Global configuration and settings.
- `models/`: Pre-trained ML models (`.pkl`).
- `services/`: Core prediction engine and feature engineering.
- `scripts/`: Training pipelines and data preparation scripts.
- `data/`: Raw and processed datasets for training.

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- Virtual Environment (recommended)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ThanhDat311/E-commerce-AI.git
   cd E-commerce-AI
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API**:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8001/api/v1`.

## 🔒 Authentication
All API endpoints require an API Key passed in the header:
- **Header Key**: `X-API-KEY`
- **Default Value**: `secret-ecommerce-ai-key-2026` (Configure in `core/config.py` or `.env`)

## 📡 API Endpoints

### Login Risk Assessment
- **POST** `/predict-login-risk`
- **Payload**:
  ```json
  {
    "user_id": 123,
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "device_type": "desktop",
    "country": "US",
    "rtt_ms": 45
  }
  ```

### Transaction Fraud Detection
- **POST** `/predict-transaction-fraud`
- **Payload**:
  ```json
  {
    "user_id": 123,
    "order_id": 456,
    "total_amount": 250.50,
    "payment_method": "credit card",
    "product_category": "electronics",
    "customer_age": 30
  }
  ```

## 🧠 Training New Models
Scripts for training are located in the `scripts/` directory. You can retrain models by running scripts like `scripts/train_transaction_fraud_model.py` after updating the dataset paths.

---
Developed for advanced E-commerce security.
