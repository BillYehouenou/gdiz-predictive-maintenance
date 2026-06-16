# ─── Variables ────────────────────────────────────────────────────────────────
PROJECT_ROOT := $(shell pwd)
PYTHONPATH   := $(PROJECT_ROOT)
UV_RUN       := PYTHONPATH=$(PYTHONPATH) uv run
PORT         ?= 8502

.PHONY: help install install-dev generate-data train pipeline \
        check lint format test \
        serve dashboard mlflow-ui \
        docker-up docker-down \
        clean clean-hard

# ─── Default target ───────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

help:  ## Affiche ce message d'aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────────────
install:  ## Installe les dépendances de production
	uv sync --frozen

install-dev:  ## Installe toutes les dépendances (dont dev/test)
	uv sync --frozen --group dev

# ─── Données & Entraînement ───────────────────────────────────────────────────
generate-data:  ## Génère le dataset synthétique GDIZ (DuckDB + Parquet)
	$(UV_RUN) python scripts/generate_synthetic_data.py

train:  ## Entraîne le modèle LightGBM et enregistre l'expérience MLflow
	$(UV_RUN) python scripts/run_training.py

pipeline: generate-data train  ## Régénère les données puis entraîne le modèle (CI complet)

# ─── Qualité du code ──────────────────────────────────────────────────────────
lint:  ## Vérifie le style du code (ruff)
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

format:  ## Formate le code automatiquement (ruff)
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

test:  ## Lance la suite de tests avec coverage
	$(UV_RUN) pytest tests/ -v --cov=src --cov=app --cov-report=term-missing

check: lint test  ## Lint + tests — gate CI avant merge

# ─── Serveurs locaux ──────────────────────────────────────────────────────────
serve:  ## Démarre l'API FastAPI en mode développement (port 8000)
	$(UV_RUN) uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dashboard:  ## Démarre le dashboard Streamlit (port $(PORT), surchargeable avec PORT=8503)
	$(UV_RUN) streamlit run dashboard.py --server.port $(PORT)

mlflow-ui:  ## Ouvre l'UI MLflow en lecture (port 5000)
	$(UV_RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-up:  ## Build et démarre tous les services (API + MLflow + dashboard)
	docker compose up --build -d

docker-down:  ## Arrête et supprime les conteneurs
	docker compose down

# ─── Nettoyage ────────────────────────────────────────────────────────────────
clean:  ## Supprime les fichiers temporaires Python
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .coverage htmlcov

clean-hard: clean  ## Supprime aussi les artefacts générés (modèles, MLflow, données)
	rm -f models/*.pkl models/*.json
	rm -f mlflow.db
	rm -rf mlruns
	rm -f data/raw/*.parquet
	@echo "Artefacts supprimés — relancer 'make pipeline' pour régénérer"
