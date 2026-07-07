from pydantic import BaseModel, Field
from typing import Optional

class TransactionInput(BaseModel):
    """
    Structure d'une transaction bancaire entrante.
    Les features V1-V28 sont les composantes PCA anonymisées.
    """
    # Features PCA
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

    # Features brutes
    Amount: float = Field(gt=0, description="Montant de la transaction en euros")

    # Features engineerées temporelles
    Hour: float = Field(ge=0, lt=24, description="Heure de la transaction (0-23)")
    sin_hour: float
    cos_hour: float
    is_night: int = Field(ge=0, le=1, description="1 si entre 22h et 6h")

    # Features engineerées Amount
    log_amount: float
    amount_normalized: float
    amount_x_night: float

    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
                "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
                "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
                "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
                "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
                "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
                "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
                "V28": -0.021053, "Amount": 149.62,
                "Hour": 0.0, "sin_hour": 0.0, "cos_hour": 1.0,
                "is_night": 1, "log_amount": 5.011, 
                "amount_normalized": 0.244, "amount_x_night": 149.62
            }
        }


class PredictionOutput(BaseModel):
    """
    Structure de la réponse de l'API.
    """
    transaction_id: str
    is_fraud: bool
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: str  # LOW / MEDIUM / HIGH / CRITICAL
    cached: bool = False
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    threshold: float