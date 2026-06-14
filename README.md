# GDIZ Maintenance Prédictive

## 1. Enjeux métiers
L'objectif principal est de réduire les temps d'arrêt non planifiés dans l'usine de la GDIZ en prédisant les pannes avant qu'elles ne surviennent. Ce projet permet de déployer une API opérationnelle pour un modèle ML capable de prédire les pannes machines. Il combine **FastAPI** pour l'inférence et la portabilité de **Docker**.

## 2. Architecture technique
Le projet suit un cycle MLOps :
1. Un lab d'entraînement : Notebooks, MLflow (Tracking) et Modèles (Artifacts).
2. Une solution de déploiement : FastAPI, Docker.
3. Un pipeline CI : GitHub Actions pour automatiser les tests et le déploiement.
4. Un système de monitoring : Suivi des prédictions avec MLflow (Logging).

## 3. Lancer le projet en une commande
Grâce à Docker, vous n'avez pas besoin d'installer Python ou les dépendances localement. Pour démarrer l'API de maintenance en une seule ligne :

````bash
docker run -d -p 8000:8000 --name gdiz-maintenance $(docker build -q .)
````
*(Cette commande construit l'image en silence et lance le conteneur sur le port 8000)*

Accès rapide :
- Documentation interactive Swagger : http://localhost:8000/docs
- Monitoring MLflow : http://localhost:5000