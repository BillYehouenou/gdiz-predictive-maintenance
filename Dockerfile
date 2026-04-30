# On utilise une image python légère
FROM python:3.11-slim

# Installation de uv et uvx pour gérer les dépendances et le lancement de l'application
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copie des fichiers de config de l'application dans le container
COPY pyproject.toml uv.lock ./

# Installation des dépendances sans créer de venv 
RUN uv sync --frozen --no-install-project

# Copie du code de l'application dans le container
COPY . .

# Expose le port sur lequel l'application va tourner
EXPOSE 8000

# Démarrage de l'application avec uvicorn
CMD ["uv run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]