from fastapi import APIRouter, Depends, HTTPException
from functools import lru_cache
from services.prediction_service import PredictionService, LoginPredictionRequest, TransactionPredictionRequest
from api.security import get_api_key

router = APIRouter()

# Singleton — models are loaded from disk ONCE when the server starts,
# then reused for every subsequent request (massive performance improvement).
@lru_cache(maxsize=1)
def get_prediction_service():
    return PredictionService()

@router.get("/health", summary="Health check — model load status")
def health_check(
    service: PredictionService = Depends(get_prediction_service)
):
    """Returns whether both AI models are loaded and ready."""
    return {
        "status": "ok",
        "models": {
            "login_risk":          service.login_model is not None,
            "transaction_fraud":   service.transaction_model is not None,
            "transaction_encoders": service.transaction_encoders is not None,
        }
    }

@router.post("/predict-login-risk", summary="Predict login risk score")
async def predict_login_risk(
    request: LoginPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
    api_key: str = Depends(get_api_key)
):
    print(f"\n[AI-API] Incoming Login Risk Request: {request.dict()}")
    try:
        result = service.predict_login_risk(request)
        return {"status": "success", "data": result}
    except Exception as e:
        print(f"[AI-API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-transaction-fraud", summary="Predict transaction fraud risk")
async def predict_transaction_fraud(
    request: TransactionPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
    api_key: str = Depends(get_api_key)
):
    print(f"\n[AI-API] Incoming Transaction Fraud Request: {request.dict()}")
    try:
        result = service.predict_transaction_fraud(request)
        return {"status": "success", "data": result}
    except Exception as e:
        print(f"[AI-API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
