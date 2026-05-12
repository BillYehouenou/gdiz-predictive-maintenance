# Utilisation d'une image Python stable et légère
FROM python:3.11-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Installation de uv pour une gestion rapide des dépendances
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copie des fichiers de configuration
COPY pyproject.toml uv.lock ./

# Installation des dépendances (sans le projet lui-même pour optimiser le cache)
RUN uv sync --frozen --no-cache

# Copie de tout le code source et des dossiers nécessaires
COPY . .

# Exposition du port FastAPI
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]