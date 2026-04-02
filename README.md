<div align="center">
  <h1>🧠 AI System: Login Risk & Payment Fraud Detection</h1>
  <p>
    <strong>A separate microservice that provides Machine Learning APIs. It connects to the Laravel website to improve security.</strong>
  </p>
</div>

---

## 📖 About the AI System
This part solves 2 main AI problems:
1. **Login Risk Assessment (RBA):** It checks login risk based on behavior (like IP, device, location, and time). It decides to: Allow login (`allow`), Ask for an OTP code (`challenge_otp`), or Block access (`block_access`).
2. **Transaction Fraud Detection:** It checks purchases to find scams in real-time. It looks at the price, account age, and time. It uses the XGBoost model and the SMOTE algorithm to balance the data.

## 💻 Technology Used
- **API Server:** FastAPI (Fast and asynchronous)
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Working:** Pandas, NumPy
- **Model Saving:** Joblib
- **Security:** API Key

---

## 🚀 How to Install and Run

### What You Need:
- Python >= 3.9

### Setup Steps for the AI System:

**Step 1: Create a Virtual Environment**
Open your terminal inside the `E-commerce-AI` folder and run this:
```bash
python -m venv venv
```

Start the environment:
- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

**Step 2: Install Python Libraries**
We removed the old libraries to save space. Install them again from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

**Step 3: Start the AI Server**
Run the server on port 8001:
```bash
python main.py
```
*(The FastAPI Server will run at `http://localhost:8001`)*

---

## 🔐 API Security (Authentication)
All API requests need a secret API Key in the Header. The Laravel website automatically sends this key:
- **Header Key**: `X-API-KEY`
- **Default Password**: `secret-ecommerce-ai-key-2026`

---

## 📂 Folder Overview
- `api/`: Holds the FastAPI routes and controllers.
- `models/`: Holds the trained models (like `.pkl` files).
- `services/`: Holds the ML logic and feature making code.
- `scripts/`: Holds the scripts to train new models from data.
- `data/`: Holds the basic CSV files for training models.
