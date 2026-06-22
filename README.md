# Maintenance Prédictive Industrielle GDIZ

Plateforme MLOps de bout en bout pour la prédiction de pannes en temps réel sur des machines textiles de la Zone Industrielle de Glo-Djigbé, Bénin.


## Résumé

Cette plateforme opérationnalise un modèle LightGBM de prédiction de pannes entraîné sur 1,05 million de relevés IoT (30 machines × 730 jours × 48 pas de 30 minutes). Elle expose une intelligence de maintenance actionnable via un dashboard Streamlit adossé à un moteur analytique DuckDB embarqué, avec MLflow gérant l'intégralité du cycle de vie des expériences. Le système est conçu pour les contraintes opérationnelles spécifiques du parc textile : déséquilibre de classes extrême (~0,02 % de taux de panne), instabilité du réseau SBEE et cycles de poussière Harmattan.


## Architecture

```
scripts/generate_synthetic_data.py   - Données IoT synthétiques (framework AI4I 2020, adapté GDIZ)
        │
        │
data/raw/gdiz_maintenance.db         - DuckDB : 1,05M lignes, indexé sur (machine_id, timestamp)
        │
src/features.py  (rolling temporel)  src/preprocessor.py  (OneHotEncoder + ColumnTransformer)
        │                                     │
        └─────────────────┬───────────────────┘
                          │
              src/train.py  +  MLflow Registry (alias @champion)
                          │     ↳ promotion gatée : la nouvelle version ne remplace le
                          │       @champion que si elle l'égale ou le dépasse sur F2
                          │
              models/model_pipeline.pkl + models/optimal_threshold.json
                          │
          ┌───────────────┴───────────────┐
          │                               │
  app/main.py (FastAPI)          dashboard.py (Streamlit)
  POST /api/v1/predict           ui/tabs/{general, machine, about}
```

*Frontière de décision :* les probabilités brutes LightGBM sont normalisées par logit de sorte que le seuil F2-optimal (0,015) corresponde exactement à 50 % affiché. Niveaux de risque UI : Normal < 35 % · Surveillance 35–50 % · Risque élevé ≥ 50 %.


## Installation 

Prérequis : Python 3.11+, uv

```bash
# Clone et installe les dépendances
git clone <url-du-repo>
cd gdiz-predictive-maintenance
uv sync --frozen

# Génère le dataset synthétique (une seule fois)
make generate-data

# Entraîne le modèle et l'enregistre dans MLflow
make train

# Dashbord Streamlit
make dashboard

# Optionnel : API d'inférence ou MLFlow UI
make serve ; make mlflow-ui
```


## Déploiement Docker

La stack Compose orchestre trois services dans des conteneurs — avec un séquençage par health-gate et exécution en utilisateur non-root :

```bash
# Placer les artefacts pré-entraînés dans models/ et les données dans data/ avant le build.
docker compose up --build -d # Démarre les services en arrière-plan
docker compose down # Arrête et supprime les conteneurs
```

L'image Docker ne contient ni données d'entraînement ni artefacts de modèle.


## Qualité du code

```bash
make lint         # ruff check + format --check + mypy (src/, app/)
make test         # pytest avec couverture (≈68 % sur src/ + app/)
make check        # lint + tests — gate CI avant merge
make check-drift  # compare la distribution des features en prod à l'entraînement (KS-test)
```

Les tests couvrent : chargement du modèle, contrat de sortie des prédictions, robustesse aux colonnes manquantes, détection de surchauffe, tous les endpoints FastAPI, le split temporel et le seuil F2 d'entraînement, la gate de promotion `@champion` et le chargement des données brutes.

*GitHub Actions* : lint + mypy - tests avec génération de données et entraînement à la volée (mis en cache entre les runs) - build Docker S'exécute sur chaque push/PR vers la branche main.


## Structure du projet

```
gdiz-predictive-maintenance/
├── app/                    # Application FastAPI
│   ├── main.py             # Routes : /health, /api/v1/predict
│   ├── schemas.py          # Modèles Pydantic entrée/sortie
│   └── utils.py            # Orchestration des prédictions + log MLflow monitoring
├── config/config.yaml      # Chemins, spec dataset, URIs MLflow
├── dashboard.py            # Point d'entrée Streamlit
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── run_training.py
│   └── check_drift.py      # Détection de drift (KS-test) — features prod vs entraînement
├── src/                    # Bibliothèque ML principale
│   ├── configloader.py
│   ├── dataloader.py
│   ├── features.py         # Feature engineering temporel (entraînement + inférence)
│   ├── preprocessor.py     # Wrapper sklearn ColumnTransformer
│   ├── train.py            # Pipeline d'entraînement MLflow + gate de promotion @champion
│   └── xpredict.py         # Predictor avec normalisation logit
├── tests/                  # Suite pytest
├── ui/                     # Modules UI Streamlit
│   ├── constants.py        # Palettes, constantes coûts métier, maps de périodes
│   ├── helpers.py          # Helpers mise en page graphiques, utilitaires de formatage
│   ├── queries.py          # Couche requêtes DuckDB 
│   ├── styles.py           # Injection du thème global CSS-in-Python
│   └── tabs/
│       ├── tab_general.py  # Onglet vue globale du parc
│       ├── tab_machine.py  # Onglet analyse par machine
│       └── tab_about.py    # Onglet architecture du pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```
