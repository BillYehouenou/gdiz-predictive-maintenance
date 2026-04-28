# API de maintenance prédictive

Ce projet déploie un modèle ML capable de prédire les pannes machines en temps réel. Il combine **FastAPI** pour l'inférence et la portabilité de **Docker**.

## Fonctionnalités
* **IA Inside** : Modèle ?? entraîné pour détecter les défaillances.
* **Validation de données** : Utilisation de Pydantic pour garantir l'intégrité des entrées capteurs.
* **Architecture ML Ops** : Tests unitaires automatisés et conteneurisation Docker.
* **Monitoring** : Interface Swagger/Streamlit.


## Installation et lancement

Le projet est entièrement conteneurisé. Aucune installation Python locale n'est requise.

1. **Build de l'image :** `docker build -t maintenance-api .`

2. **Lancement du conteneur :** `docker run -p 8000:8000 maintenance-api`

3. **Accès à l'API :**
   - Endpoint de prédiction : `http://localhost:8000/predict`
   - Documentation interactive : `http://localhost:8000/docs`

## Tests

Des tests unitaires sont inclus pour valider les prédictions de l'API. Ils peuvent être exécutés localement ou via GitHub Actions.

```bash
python -m pytest
```
