from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
import uuid
import time

from schemas import TransactionInput, PredictionOutput, HealthResponse
from predictor import predictor, API_ERRORS


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_model()
    predictor.connect_redis()  # Plus de paramètres — lit depuis env vars
    yield
    print("API arrêtée")


app = FastAPI(
    title="🔒 Fraud Detection API",
    description="""
    API de détection de fraude bancaire en temps réel.

    - **Modèle** : XGBoost + SMOTE
    - **Seuil optimal** : 0.96
    - **Cache** : Redis (TTL 1h)
    - **F2-Score** : 0.817 | **Precision** : 94.9% | **Recall** : 79.0%
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Instrumentation automatique Prometheus
# Expose /metrics avec métriques HTTP standard (requêtes, latence, erreurs)
Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None,
        redis_connected=predictor.redis_available,
        threshold=predictor.threshold or 0.0
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_fraud(transaction: TransactionInput):
    if predictor.model is None:
        API_ERRORS.labels(error_type='model_not_loaded').inc()
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé — API non prête"
        )

    transaction_id = str(uuid.uuid4())

    try:
        result = predictor.predict(
            transaction.model_dump(),
            transaction_id
        )
    except Exception as e:
        API_ERRORS.labels(error_type='prediction_error').inc()
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionOutput(**result)


@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }