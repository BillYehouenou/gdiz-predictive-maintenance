.PHONY: install install-dev lint format test train serve mlflow-ui docker-up docker-down clean

install:
	uv sync --frozen

install-dev:
	uv sync --frozen --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest tests/ -v --cov=src --cov=app --cov-report=term-missing

train:
	uv run python scripts/run_training.py

serve:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .coverage htmlcov
