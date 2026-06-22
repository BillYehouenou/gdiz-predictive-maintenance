# Utilisation d'une image Python stable et légère
FROM python:3.11-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# curl : requis par les healthchecks docker-compose (mlflow/api/dashboard).
# libgomp1 : runtime OpenMP requis par LightGBM, absent de python:3.11-slim.
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && rm -rf /var/lib/apt/lists/*

# Installation de uv pour une gestion rapide des dépendances
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copie des fichiers de configuration
COPY pyproject.toml uv.lock ./

# Installation des dépendances (sans le projet lui-même pour optimiser le cache)
RUN uv sync --frozen --no-cache

# Copie de tout le code source et des dossiers nécessaires
COPY . .

# Exécution en utilisateur non-root (hardening) — /app/mlflow accueille le volume
# nommé mlflow_data du service mlflow, créé ici pour qu'il hérite du bon propriétaire.
RUN mkdir -p /app/mlflow && \
    useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exposition du port FastAPI
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]