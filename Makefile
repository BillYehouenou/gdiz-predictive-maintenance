UV_RUN := PYTHONHASHSEED=0 PYTHONPATH=$(CURDIR) uv run

.PHONY: help demo install generate-data mlflow train score pipeline check-drift lint format test check serve docker-up docker-down clean
.DEFAULT_GOAL := help

help:  ## Affiche les commandes disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

demo:  ## Lance tout sans Docker (registre, données et modèle si besoin) et ouvre le tableau de bord
	$(UV_RUN) python -m src.demarrer

install:  ## Installe les dépendances (dont celles de développement)
	uv sync --frozen

# Données, registre et modèle
generate-data:  ## Génère le dataset (météo réelle Open-Meteo + simulation des machines --> DuckDB)
	$(UV_RUN) python -m simulation.generer

mlflow:  ## Démarre le registre MLflow (suivi des entraînements, @champion) sur http://localhost:5050
	docker compose up -d --wait mlflow

train: mlflow  ## Entraîne le LightGBM et l'enregistre dans MLflow (@champion si le recall ne régresse pas)
	$(UV_RUN) python -m src.train

score: mlflow  ## Score les 60 derniers jours avec le @champion --> table predictions
	$(UV_RUN) python -m src.score

pipeline: generate-data train score  ## Régénère les données, entraîne, score

check-drift:  ## Compare les 30 derniers jours de capteurs à la période d'entraînement (PSI)
	$(UV_RUN) python -m src.drift

# Qualité
lint:  ## Style et types (ruff + mypy)
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .
	$(UV_RUN) mypy src app simulation

format:  ## Formate le code (ruff)
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

test: mlflow  ## Tests avec couverture
	$(UV_RUN) pytest tests/ --cov=src --cov=app --cov-report=term-missing

check: lint test  ## Lint + tests

# Serveurs
serve: mlflow  ## API et tableau de bord sur http://localhost:8000
	$(UV_RUN) uvicorn app.main:app --reload --reload-dir app --reload-dir src --reload-dir web --host 0.0.0.0 --port 8000

docker-up:  ## Tout le projet dans Docker : registre, données et modèle si besoin, tableau de bord sur :8000
	docker compose up --build -d --wait
	@echo "Tableau de bord sur http://localhost:8000"

docker-down:  ## Arrête la stack (données et registre sont conservés dans leurs volumes)
	docker compose down

clean:  ## Supprime les caches Python
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
