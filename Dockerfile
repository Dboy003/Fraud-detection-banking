# Image de base — Python 3.11 slim (légère)
FROM python:3.11-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Installation package par package pour contrôler exactement ce qui est installé
RUN pip install --no-cache-dir fastapi==0.115.0
RUN pip install --no-cache-dir uvicorn==0.30.0
RUN pip install --no-cache-dir redis==5.0.8
RUN pip install --no-cache-dir pydantic==2.10.0
RUN pip install --no-cache-dir joblib==1.4.2
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir pandas==2.2.2
RUN pip install --no-cache-dir scikit-learn==1.5.2
RUN pip install --no-cache-dir scipy==1.13.1
RUN pip install --no-cache-dir imbalanced-learn==0.12.3
RUN pip install --no-cache-dir --no-deps xgboost==2.1.0
RUN pip install --no-cache-dir prometheus-client==0.20.0
RUN pip install --no-cache-dir prometheus-fastapi-instrumentator==7.0.0

# Copie du code de l'API
COPY api/ ./api/
COPY models/ ./models/

# Ajoute /app/api au chemin de recherche des modules Python
ENV PYTHONPATH=/app/api

# Port exposé
EXPOSE 8000

# Commande de démarrage
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]