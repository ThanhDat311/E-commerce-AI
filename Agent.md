# Agent AI Assistant Guidelines: E-commerce Fraud Detection API

## 1. Project Context
This is a standalone Python **Microservice** built with **FastAPI** (`E-commerce-AI`). Core purpose is to provide Machine Learning prediction endpoints for an external **Laravel E-commerce application**. The Laravel app is the primary source of truth for the database and handles user interactions. This API strictly handles data processing and AI inference.

## 2. Core Constraints & Architecture
- **Language & Framework:** Python 3.x, FastAPI.
- **Server Deployment:** ASGI server, usually `uvicorn`. Standard development port is **8001** (changed from 8000 to avoid conflicts with Laragon/Apache).
- **Stateless Design:** This API should ideally be stateless. All required context for a prediction must be passed in the JSON payload of the HTTP request from the Laravel frontend.
- **Data Persistence:** This project *does not* own the primary database (MySQL). It operates on datasets (like CSV files or read-only DB dumps) exclusively for **training** models. Operational predictions do not require writing back to a local database here.

## 3. Core Domains

### 3.1. Login Risk Assessment (Authentication)
- **Goal:** Assess the risk of Account Takeover (ATO) or malicious login attempts.
- **Input Data Scope:** IP Address, User Agent, Location, Device Fingerprint, Timestamp, Round-Trip Time.
- **Related Kaggle Resource:** RBA Dataset (Risk-Based Authentication).
- **Endpoint:** `POST /api/v1/predict-login-risk`
- **Output:** Returns a `risk_score` (0.0 - 1.0) and an `auth_decision` (e.g., `allow`, `challenge_otp`, `block`).

### 3.2. Transaction Fraud Detection (Purchasing)
- **Goal:** Analyze orders at checkout to prevent credit card fraud and anomalous purchasing behavior.
- **Input Data Scope:** User profile data, Total Amount, Payment Method, Product Categories, Order velocity.
- **Related Kaggle Resource:** E-commerce Fraudulent Transactions Dataset.
- **Endpoint:** `POST /api/v1/predict-transaction-fraud`
- **Output:** Returns a `risk_score` (0.0 - 1.0) and a `decision` (e.g., `allow`, `review`, `block`).

## 4. Workflows

### 4.1. Training a Model
1. Place the Kaggle datasets (CSV/JSON) into a `data/raw/` directory.
2. Create Jupyter notebooks or Python scripts in a `scripts/` or `notebooks/` directory to perform Exploratory Data Analysis (EDA) and Data Preprocessing.
3. Train the model using libraries like `scikit-learn` or `xgboost`.
4. Serialize (save) the trained model using `joblib` into the `models/` directory (e.g., `models/xgboost_login_model_v1.pkl`).

### 4.2. Updating the API for Inference
1. Load the serialized `.pkl` model in the `__init__` method of `services/prediction_service.py` to ensure it only loads into memory once when the API starts.
2. In the prediction methods, write code to transform the incoming Pydantic `BaseModel` request payload into the exact feature array shape (`numpy` array or `pandas` DataFrame) that the model expects.
3. Call `model.predict_proba()` to extract the risk probability.

## 5. Integration with Laravel (Communication Protocol)
- **Transport:** Standard HTTP REST API.
- **Format:** JSON.
- **Laravel Implementation:** On the PHP side, the Laravel application uses the Guzzle HTTP Client to make synchronous or asynchronous POST requests to `http://localhost:8001/api/v1/...` during specific events (like `AttemptingLogin` or `CheckoutProcessed`).
- **Error Handling:** Should this API go down, Laravel must have fallback logic (e.g., default to `allow` with a warning, or queue the check) to avoid locking users out. This API must return clean HTTP 5xx errors if internal exceptions occur.

## 6. Project Structure Overview
- `main.py`: The FastAPI application instance and uvicorn runner.
- `api/routes.py`: Defines the HTTP endpoints and maps them to service functions.
- `services/prediction_service.py`: Contains the core business logic, model loading, feature extraction, and prediction algorithms.
- `core/config.py`: Pydantic settings management for environment variables and file paths.
- `models/`: Directory holding compiled `.pkl` files.
- `requirements.txt`: Python pip dependencies.

## 7. Development Guidelines
- Always use **Type Hints**. FastAPI relies heavily on Pydantic typings for request validation and Swagger doc generation.
- Mocking: For endpoints where a model is not yet trained, return mock `random` data strictly conforming to the defined output schema so the Laravel client can continue development unblocked.
