import joblib
import json
import hashlib
import numpy as np
import pandas as pd
import redis
import os
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, Summary

# ─── Chemins ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_detection_pipeline.pkl')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_config.json')

FEATURE_COLUMNS = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount','Hour','sin_hour','cos_hour','is_night',
    'log_amount','amount_normalized','amount_x_night'
]

# ─── Métriques Prometheus ───────────────────────────────────────────────────

# Nombre total de prédictions
PREDICTIONS_TOTAL = Counter(
    'fraud_predictions_total',
    'Nombre total de prédictions',
    ['result']  # label: fraud / legitimate
)

# Latence des prédictions (p50, p95, p99)
PREDICTION_LATENCY = Histogram(
    'fraud_prediction_latency_seconds',
    'Latence des prédictions en secondes',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]
)

# Hit rate Redis
CACHE_HITS = Counter(
    'fraud_cache_hits_total',
    'Nombre de réponses servies depuis le cache Redis'
)
CACHE_MISSES = Counter(
    'fraud_cache_misses_total',
    'Nombre de prédictions calculées par le modèle'
)

# Taux de fraude (gauge = valeur instantanée)
FRAUD_RATE = Gauge(
    'fraud_rate_current',
    'Taux de fraude sur les 100 dernières transactions'
)

# Distribution des probabilités de fraude
FRAUD_PROBABILITY = Histogram(
    'fraud_probability_distribution',
    'Distribution des probabilités de fraude prédites',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.96, 1.0]
)

# Distribution par niveau de risque
RISK_LEVEL_COUNTER = Counter(
    'fraud_risk_level_total',
    'Distribution des niveaux de risque',
    ['level']  # LOW / MEDIUM / HIGH / CRITICAL
)

# Erreurs API
API_ERRORS = Counter(
    'fraud_api_errors_total',
    'Nombre d erreurs API',
    ['error_type']
)

# Data drift — moyenne des features V par fenêtre
FEATURE_MEAN = Gauge(
    'fraud_feature_mean',
    'Moyenne glissante des features V (data drift)',
    ['feature']
)

# Prediction drift — score de confiance moyen
PREDICTION_CONFIDENCE = Gauge(
    'fraud_prediction_confidence_mean',
    'Score de confiance moyen des 100 dernières prédictions'
)


class FraudPredictor:
    def __init__(self):
        self.model = None
        self.threshold = None
        self.redis_client = None
        self.redis_available = False
        # Fenêtre glissante pour les métriques drift
        self._recent_predictions = []
        self._recent_probabilities = []
        self._window_size = 100

    def load_model(self):
        self.model = joblib.load(MODEL_PATH)
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        self.threshold = config['optimal_threshold']
        print(f"Modèle chargé — seuil optimal : {self.threshold}")

    def connect_redis(self):
        """Connexion à Redis — paramètres lus depuis variables d'environnement.

        Compatible Redis local (Docker Compose, sans mot de passe/TLS)
        et Upstash Redis (mot de passe + TLS obligatoires).
        """
        host = os.getenv('REDIS_HOST', 'localhost')
        port = int(os.getenv('REDIS_PORT', 6379))
        password = os.getenv('REDIS_PASSWORD', None)
        use_ssl = os.getenv('REDIS_SSL', 'false').lower() == 'true'

        try:
            self.redis_client = redis.Redis(
                host=host, port=port, db=0,
                password=password,
                ssl=use_ssl,
                socket_connect_timeout=5,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_available = True
            print(f"Redis connecté — {host}:{port} (SSL={use_ssl})")
        except Exception as e:
            self.redis_available = False
            print(f"Redis indisponible — mode sans cache : {e}")

    def _get_cache_key(self, transaction_data: dict) -> str:
        transaction_str = json.dumps(transaction_data, sort_keys=True)
        return f"fraud:{hashlib.md5(transaction_str.encode()).hexdigest()}"

    def _get_risk_level(self, probability: float) -> str:
        if probability < 0.3:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        elif probability < 0.9:
            return "HIGH"
        else:
            return "CRITICAL"

    def _update_drift_metrics(self, transaction_data: dict, probability: float):
        """
        Met à jour les métriques de data drift et prediction drift
        sur une fenêtre glissante de 100 transactions.
        """
        # Fenêtre glissante des probabilités
        self._recent_probabilities.append(probability)
        if len(self._recent_probabilities) > self._window_size:
            self._recent_probabilities.pop(0)

        # Prediction drift — score de confiance moyen
        PREDICTION_CONFIDENCE.set(np.mean(self._recent_probabilities))

        # Data drift — moyenne des features V1-V28
        for feature in [f'V{i}' for i in range(1, 29)]:
            if feature in transaction_data:
                # Fenêtre glissante par feature
                FEATURE_MEAN.labels(feature=feature).set(
                    transaction_data[feature]
                )

        # Taux de fraude sur fenêtre glissante
        self._recent_predictions.append(1 if probability >= self.threshold else 0)
        if len(self._recent_predictions) > self._window_size:
            self._recent_predictions.pop(0)
        FRAUD_RATE.set(np.mean(self._recent_predictions))

    def predict(self, transaction_data: dict, transaction_id: str) -> dict:
        import time
        cache_key = self._get_cache_key(transaction_data)

        # 1. Vérification cache Redis
        if self.redis_available:
            cached = self.redis_client.get(cache_key)
            if cached:
                CACHE_HITS.inc()
                result = json.loads(cached)
                result['transaction_id'] = transaction_id
                result['cached'] = True
                # Métriques même sur cache hit
                PREDICTIONS_TOTAL.labels(
                    result='fraud' if result['is_fraud'] else 'legitimate'
                ).inc()
                RISK_LEVEL_COUNTER.labels(level=result['risk_level']).inc()
                return result

        CACHE_MISSES.inc()

        # 2. Préparation features
        df = pd.DataFrame([transaction_data])[FEATURE_COLUMNS]

        # 3. Prédiction avec mesure de latence
        start = time.time()
        fraud_probability = float(self.model.predict_proba(df)[:, 1][0])
        latency = time.time() - start
        PREDICTION_LATENCY.observe(latency)

        is_fraud = fraud_probability >= self.threshold
        risk_level = self._get_risk_level(fraud_probability)

        # 4. Mise à jour métriques
        PREDICTIONS_TOTAL.labels(
            result='fraud' if is_fraud else 'legitimate'
        ).inc()
        FRAUD_PROBABILITY.observe(fraud_probability)
        RISK_LEVEL_COUNTER.labels(level=risk_level).inc()
        self._update_drift_metrics(transaction_data, fraud_probability)

        # 5. Construction réponse
        result = {
            'transaction_id': transaction_id,
            'is_fraud': bool(is_fraud),
            'fraud_probability': round(fraud_probability, 4),
            'risk_level': risk_level,
            'cached': False,
            'message': (
                '⚠️ Transaction frauduleuse détectée'
                if is_fraud
                else '✅ Transaction légitime'
            )
        }

        # 6. Mise en cache Redis
        if self.redis_available:
            cache_value = {k: v for k, v in result.items()
                          if k != 'transaction_id'}
            self.redis_client.setex(
                cache_key, 3600, json.dumps(cache_value)
            )

        return result


predictor = FraudPredictor()