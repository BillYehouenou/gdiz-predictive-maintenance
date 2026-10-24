# Variables d'environnement
PROJECT_ROOT := $(shell pwd)
PYTHONPATH   := $(PROJECT_ROOT)
UV_RUN       := PYTHONHASHSEED=0 PYTHONPATH=$(PYTHONPATH) uv run

.PHONY: help install install-dev generate-data train compare score interventions pipeline check-drift \
        check lint format test \
        serve mlflow-ui \
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

train:  # Entraîne le modèle retenu et l'enregistre dans MLflow
	$(UV_RUN) python -m src.train

compare:  # Compare les modèles candidats (validation sur 3 fenêtres + test 2025)
	$(UV_RUN) python -m src.modeles

score:  # Score tout le parc avec le @champion --> table predictions (lue par l'API)
	$(UV_RUN) python -m src.score

interventions:  # (Re)crée la base des interventions : historique 2025 reconstitué + alertes en cours (API arrêtée)
	$(UV_RUN) python -m src.interventions

pipeline: generate-data train score interventions  # Régénère les données, entraîne, score le parc, prépare les interventions

check-drift:  # Compare les 30 derniers jours (features et risque prédit) à la période d'entraînement
	$(UV_RUN) python -m src.drift

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
serve:  # Démarre l'API et le tableau de bord sur http://localhost:8000
	$(UV_RUN) uvicorn app.main:app --reload --reload-dir app --reload-dir src --reload-dir web --host 0.0.0.0 --port 8000

mlflow-ui:  # Ouvre l'UI MLflow en lecture 
	$(UV_RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Docker
docker-up:  # Build et démarre tous les services (API et tableau de bord + MLflow)
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
