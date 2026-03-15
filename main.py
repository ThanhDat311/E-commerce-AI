from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from api.routes import router as api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Microservice providing AI Risk Assessment and Fraud Detection for E-commerce",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Setup - update to restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", summary="Root Access")
async def root():
    return {
        "message": "Welcome to E-commerce Fraud Detection AI API",
        "status": "healthy"
    }

@app.get("/health", summary="Health Check")
async def get_health():
    return {
        "message": "E-commerce Fraud Detection AI API is running",
        "status": "healthy"
    }

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8001 to avoid Laragon Apache conflict
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
