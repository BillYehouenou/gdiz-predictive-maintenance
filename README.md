# Maintenance Prédictive Industrielle GDIZ

Plateforme MLOps de bout en bout pour la prédiction de pannes en temps réel sur des machines textiles de la Zone Industrielle de Glo-Djigbé, Bénin.


## Résumé

Cette plateforme opérationnalise un modèle LightGBM de prédiction de pannes entraîné sur 1,75 million de relevés IoT (50 machines × 730 jours × 48 pas de 30 minutes). Elle expose une intelligence de maintenance actionnable via un dashboard Streamlit adossé à un moteur analytique DuckDB embarqué, avec MLflow gérant l'intégralité du cycle de vie des expériences. Le système est conçu pour les contraintes opérationnelles spécifiques du parc textile : déséquilibre de classes extrême, instabilité du réseau SBEE et cycles de poussière Harmattan.

La cible tente de répondre à la question : **cette machine tombera-t-elle en panne dans les 120h prochaines heures ? Si oui, quelle en est la cause probable ?**


## Pipeline

```
Open-Meteo (réel)                    - Météo de Glo-Djigbé 2024-2025 : température, humidité, pluie, poussière (CAMS)
        │
simulation/                          - Physique des machines (framework AI4I 2020, adapté GDIZ) --> défauts capteurs IoT
        │                              L'usure et l'encrassement restent cachés : seuls capteurs + compteurs GMAO sont exportés
        │
data/raw/gdiz_maintenance.db         - DuckDB : télémétrie 1,75M lignes (id_machine, timestamp) + cache météo
        │
src/donnees.py + src/features.py     - cible à 120h, features glissantes (24h / 4h) : une seule fonction,
        │                              identique à l'entraînement, dans le dashboard et dans l'API
        │                              split train / validation / test avec tampon de 120h + 10 machines réservées au test
        │
src/train.py                         - LightGBM (NaN et catégories natives) --> calibration sigmoïde --> seuil choisi
        │                              sur la validation (recall événementiel sous budget de fausses alertes)
        │                              --> MLflow Registry : @champion remplacé seulement si le recall ne régresse pas
        │
              models/model_pipeline.pkl + models/seuil_optimal.json
                          │
          ┌───────────────┴───────────────┐
          │                               │
  app/main.py (FastAPI)          dashboard.py (Streamlit)
  POST /api/v1/predict           ui/tabs/{general, machine, about}
  POST /api/v1/predict/batch
  GET  /api/v1/info              (version + seuil du champion chargé)
  POST /api/v1/reload            (recharge le @champion sans redémarrer)
```


## Installation 

Prérequis : Python 3.11+, uv

```bash
# Clone et installe les dépendances
git clone <url-du-repo>
cd gdiz-predictive-maintenance
uv sync --frozen

# Génère le dataset (télécharge la météo Open-Meteo au premier lancement, puis cache DuckDB)
make generate-data

# Entraîne le modèle et l'enregistre dans MLflow
make train

# Dashbord Streamlit
make dashboard

# Optionnel : API d'inférence ou MLFlow UI
make serve ; make mlflow-ui
```


## Déploiement Docker

La stack Compose orchestre trois services dans des conteneurs :

```bash
# Placer les artefacts pré-entraînés dans models/ et les données dans data/ avant le build.
docker compose up --build -d # Démarre les services en arrière-plan
docker compose down # Arrête et supprime les conteneurs
```

L'image Docker ne contient ni données d'entraînement ni artefacts de modèle.
*GitHub Actions* : lint + mypy ; tests avec génération de données et entraînement à la volée (mis en cache entre les runs) - build Docker S'exécute sur chaque push/PR vers la branche main.


## Structure du projet

```
gdiz-predictive-maintenance/
├── app/                    # Application FastAPI
│   ├── main.py             # Routes : /health, /predict(/batch), /info, /reload
│   ├── schemas.py          # Modèles Pydantic entrée/sortie
│   └── utils.py            # Orchestration des prédictions, log MLflow async, rechargement à chaud
├── dashboard.py            # Point d'entrée Streamlit
├── simulation/             # Générateur : python -m simulation.generer
│   ├── meteo.py            # Météo réelle Open-Meteo (ERA5 + CAMS), cache DuckDB
│   ├── physique.py         # Atelier (extérieur --> intérieur) + état caché et risques de panne des machines
│   ├── observation.py      # Défaillances capteurs IoT + écriture DuckDB
│   └── generer.py          # Orchestration
├── scripts/
│   ├── run_baseline_comparison.py  # Règle triviale + régression logistique, à battre par LightGBM
│   └── check_drift.py      # Drift KS + PSI des features et des prédictions
├── src/                    # Pipeline ML : python -m src.train
│   ├── config.py           # Chemins, schéma des données, MLflow
│   ├── donnees.py          # Chargement DuckDB, cible à horizon, split avec tampon et machines réservées
│   ├── features.py         # Features glissantes (une seule fonction pour l'entraînement et l'inférence)
│   ├── evaluation.py       # Métrique événementielle : recall @ budget de fausses alertes
│   ├── train.py            # LightGBM + calibration + seuil + promotion @champion
│   ├── predicteur.py       # Chargement du @champion et prédiction
│   └── explain.py          # Cause probable par famille physique (SHAP)
├── tests/                  # Suite pytest
├── ui/                     # Modules UI Streamlit
│   ├── constants.py        # Palettes, constantes coûts métier, maps de périodes
│   ├── helpers.py          # Helpers mise en page graphiques, utilitaires de formatage
│   ├── priorites.py        # Vue "à traiter aujourd'hui" — machines en alerte triées par risque
│   ├── queries.py          # Couche requêtes DuckDB 
│   ├── styles.py           # Injection du thème global CSS-in-Python
│   └── tabs/
│       ├── tab_general.py  # Onglet vue globale du parc + priorités du jour
│       ├── tab_machine.py  # Onglet analyse par machine + cause probable (SHAP)
│       └── tab_about.py    # Onglet architecture du pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```
