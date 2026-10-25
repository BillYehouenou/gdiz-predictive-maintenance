# Image de l'API et du pipeline (données, entraînement, scoring) ; les données vivent dans un volume, le modèle dans MLflow
FROM python:3.11-slim

WORKDIR /app

# curl : healthcheck de docker-compose ; libgomp1 : OpenMP, requis par LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /bin/

# Dépendances de production seules, avant le code pour profiter du cache Docker
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

COPY . .

# data/raw appartient à appuser : le volume des données en hérite à sa création
RUN mkdir -p data/raw && useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
