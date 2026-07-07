import joblib
import json
import hashlib
import numpy as np
import pandas as pd
import redis
import os
from typing import Optional

# Chemins
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_detection_pipeline.pkl')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_config.json')

# Ordre des features (doit correspondre exactement à l'entraînement)
FEATURE_COLUMNS = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount','Hour','sin_hour','cos_hour','is_night',
    'log_amount','amount_normalized','amount_x_night'
]


class FraudPredictor:
    def __init__(self):
        self.model = None
        self.threshold = None
        self.redis_client = None
        self.redis_available = False

    def load_model(self):
        """Charge le modèle et la config au démarrage de l'API."""
        # Chargement du pipeline
        self.model = joblib.load(MODEL_PATH)

        # Chargement du seuil optimal
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        self.threshold = config['optimal_threshold']

        print(f"Modèle chargé — seuil optimal : {self.threshold}")

    def connect_redis(self, host: str = 'localhost', port: int = 6379):
        """Connexion à Redis avec fallback gracieux si indisponible."""
        try:
            self.redis_client = redis.Redis(
                host=host, port=port, db=0,
                socket_connect_timeout=2,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_available = True
            print("Redis connecté")
        except Exception as e:
            self.redis_available = False
            print(f"Redis indisponible — mode sans cache : {e}")

    def _get_cache_key(self, transaction_data: dict) -> str:
        """Génère une clé unique pour chaque transaction."""
        transaction_str = json.dumps(transaction_data, sort_keys=True)
        return f"fraud:{hashlib.md5(transaction_str.encode()).hexdigest()}"

    def _get_risk_level(self, probability: float) -> str:
        """Traduit la probabilité en niveau de risque métier."""
        if probability < 0.3:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        elif probability < 0.9:
            return "HIGH"
        else:
            return "CRITICAL"

    def predict(self, transaction_data: dict, transaction_id: str) -> dict:
        """
        Prédit si une transaction est frauduleuse.
        Vérifie d'abord le cache Redis avant d'appeler le modèle.
        """
        cache_key = self._get_cache_key(transaction_data)

        # 1. Vérification du cache Redis
        if self.redis_available:
            cached = self.redis_client.get(cache_key)
            if cached:
                result = json.loads(cached)
                result['transaction_id'] = transaction_id
                result['cached'] = True
                return result

        # 2. Préparation des features
        df = pd.DataFrame([transaction_data])[FEATURE_COLUMNS]

        # 3. Prédiction
        fraud_probability = float(
            self.model.predict_proba(df)[:, 1][0]
        )
        is_fraud = fraud_probability >= self.threshold

        # 4. Construction de la réponse
        result = {
            'transaction_id': transaction_id,
            'is_fraud': bool(is_fraud),
            'fraud_probability': round(fraud_probability, 4),
            'risk_level': self._get_risk_level(fraud_probability),
            'cached': False,
            'message': (
                '⚠️ Transaction frauduleuse détectée' 
                if is_fraud 
                else '✅ Transaction légitime'
            )
        }

        # 5. Mise en cache Redis (TTL : 1 heure)
        if self.redis_available:
            cache_value = {k: v for k, v in result.items() 
                          if k != 'transaction_id'}
            self.redis_client.setex(
                cache_key, 3600, json.dumps(cache_value)
            )

        return result


# Instance globale — chargée une fois au démarrage
predictor = FraudPredictor()