from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uuid
import time

from schemas import TransactionInput, PredictionOutput, HealthResponse
from predictor import predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement du modèle et connexion Redis au démarrage."""
    predictor.load_model()
    predictor.connect_redis()
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


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Vérifie l'état de l'API, du modèle et de Redis."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None,
        redis_connected=predictor.redis_available,
        threshold=predictor.threshold or 0.0
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_fraud(transaction: TransactionInput):
    """
    Prédit si une transaction bancaire est frauduleuse.
    
    Retourne la probabilité de fraude, le niveau de risque,
    et indique si la réponse provient du cache Redis.
    """
    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé — API non prête"
        )

    # Génération d'un ID unique pour la transaction
    transaction_id = str(uuid.uuid4())

    # Prédiction
    start_time = time.time()
    result = predictor.predict(
        transaction.model_dump(),
        transaction_id
    )
    latency_ms = (time.time() - start_time) * 1000

    print(f"[{transaction_id[:8]}] "
          f"Fraude: {result['is_fraud']} | "
          f"Prob: {result['fraud_probability']:.4f} | "
          f"Latence: {latency_ms:.1f}ms | "
          f"Cache: {result['cached']}")

    return PredictionOutput(**result)


@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health"
    }