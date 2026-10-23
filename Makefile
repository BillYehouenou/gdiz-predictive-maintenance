# Variables d'environnement
PROJECT_ROOT := $(shell pwd)
PYTHONPATH   := $(PROJECT_ROOT)
UV_RUN       := PYTHONHASHSEED=0 PYTHONPATH=$(PYTHONPATH) uv run
PORT         ?= 8501

.PHONY: help install install-dev generate-data train compare pipeline check-drift \
        check lint format test \
        serve dashboard mlflow-ui \
        docker-up docker-down \
        clean clean-hard

# Définit la cible par défaut si aucune n'est spécifiée
.DEFAULT_GOAL := help

help:  ## Affiche ce message d'aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# Configuration des dépendances
install:  # installe les dépendances de production
	uv sync --frozen

install-dev:  # installe toutes les dépendances (dont ceux de l'environnement de dev)
	uv sync --frozen --group dev

# Données & Entraînement
generate-data:  # Génère le dataset GDIZ (météo réelle Open-Meteo + simulation des machines --> DuckDB)
	$(UV_RUN) python -m simulation.generer

train:  # Entraîne le modèle LightGBM et enregistre l'expérience MLflow
	$(UV_RUN) python -m src.train

pipeline: generate-data train  # Régénère les données puis entraîne le modèle

compare:  # Compare les modèles candidats (validation sur 3 fenêtres + test 2025)
	$(UV_RUN) python -m src.modeles

check-drift:  # Compare la distribution des features en production à celle de l'entraînement
	$(UV_RUN) python scripts/check_drift.py

# Qualité du code 
lint:  # vérifie le style et les types (ruff + mypy)
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .
	$(UV_RUN) mypy src app simulation

format:  # formate le code automatiquement (ruff)
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

test:  # lance la suite de tests avec coverage
	$(UV_RUN) pytest tests/ -v --cov=src --cov=app --cov-report=term-missing

check: lint test  # lint + tests : gate CI avant merge

# Serveurs locaux
serve:  # Démarre l'API FastAPI en mode développement
	$(UV_RUN) uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dashboard:  # Démarre le dashboard Streamlit 
	$(UV_RUN) streamlit run dashboard.py --server.port $(PORT)

mlflow-ui:  # Ouvre l'UI MLflow en lecture 
	$(UV_RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Docker
docker-up:  # Build et démarre tous les services (API + MLflow + dashboard)
	docker compose up --build -d

docker-down:  # Arrête et supprime les conteneurs
	docker compose down

# Nettoyage 
clean:  # Supprime les fichiers temporaires Python
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .coverage htmlcov

clean-hard: clean  # Supprime aussi les artefacts générés (modèles, MLflow, données)
	rm -f models/*.pkl models/*.json
	rm -f mlflow.db
	rm -rf mlruns
	rm -f data/raw/*.db
	@echo "Artefacts supprimés — relancer 'make pipeline' pour régénérer"
