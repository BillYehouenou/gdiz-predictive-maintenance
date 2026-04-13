# 1. On part d'une image Python officielle légère
FROM python:3.10-slim

# 2. On définit le dossier de travail dans le conteneur
WORKDIR /app

# 3. On copie le fichier des dépendances
COPY requirements.txt .

# 4. On installe les bibliothèques
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie tout le reste du code (app, src, etc.)
COPY . .

# 6. On expose le port 8000 (celui de FastAPI)
EXPOSE 8000

# 7. La commande pour lancer l'API au démarrage du conteneur
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]