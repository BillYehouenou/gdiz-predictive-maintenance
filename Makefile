UV_RUN := PYTHONHASHSEED=0 PYTHONPATH=$(CURDIR) uv run

.PHONY: help demo install generate-data mlflow train score pipeline check-drift lint format test serve docker-up docker-down
.DEFAULT_GOAL := help

help:  ## Affiche les commandes disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

demo:  ## Tout sans Docker (registre, données et modèle si besoin) et ouvre le tableau de bord
	$(UV_RUN) python -m src.demarrer

install:  ## Installe les dépendances
	uv sync --frozen

generate-data:  ## Génère le dataset (météo réelle Open-Meteo + simulation des machines --> DuckDB)
	$(UV_RUN) python -m simulation.generer

mlflow:  ## Démarre le registre MLflow sur http://localhost:5050 (Docker)
	docker compose up -d --wait mlflow

train: mlflow  ## Entraîne le LightGBM et l'enregistre (@champion si le gain net en validation ne baisse pas)
	$(UV_RUN) python -m src.train

score: mlflow  ## Score les 60 derniers jours avec le @champion --> table predictions
	$(UV_RUN) python -m src.modele

pipeline: generate-data train score  ## Données, entraînement et scoring

check-drift: mlflow  ## Performance réelle des 60 derniers jours et dérive des capteurs (même saison)
	$(UV_RUN) python -m src.drift

lint:  ## Style et types (ruff + mypy)
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .
	$(UV_RUN) mypy src app simulation

format:  ## Formate le code (ruff)
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

test: mlflow  ## Tests avec couverture
	$(UV_RUN) pytest tests/ --cov=src --cov=app --cov-report=term-missing

serve: mlflow  ## API et tableau de bord sur http://localhost:8000
	$(UV_RUN) uvicorn app.main:app --reload --reload-dir app --reload-dir src --reload-dir web --port 8000

docker-up:  ## Tout le projet dans Docker, tableau de bord sur http://localhost:8000
	docker compose up --build -d --wait

docker-down:  ## Arrête la stack (données et registre restent dans leurs volumes)
	docker compose down
