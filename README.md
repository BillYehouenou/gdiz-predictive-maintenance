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
src/score.py                         - score tout le parc une fois --> table DuckDB `predictions`
        │                              (risque, niveau, cause probable SHAP des machines en alerte)
          ┌───────────────┴───────────────┐
          │                               │
  app/main.py (FastAPI)          dashboard.py (Streamlit) : lit la table, ne charge pas le modèle
  POST /api/v1/predict(/batch)   (prédiction temps réel, sans état)
  GET  /api/v1/machines          (dernier état de chaque machine)
  GET  /api/v1/priorites         (machines en alerte, par risque décroissant)
  GET  /api/v1/machines/{id}/serie?jours=30
  GET  /api/v1/info · POST /api/v1/reload
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

# Entraîne le modèle (MLflow) puis score le parc (table predictions)
make train score

# Dashbord Streamlit
make dashboard

# Optionnel : API d'inférence ou MLFlow UI
make serve ; make mlflow-ui
```


## Choix du modèle

Recall événementiel au budget de 0,2 fausse alerte par machine et par jour (`make compare`) :

| Modèle | Validation (T2-T4 2024, 65 pannes) | Test 2025 (115 pannes) | Taille |
|---|---|---|---|
| Règle : heures d'outil | 59 % | 50 % | 0,2 Ko |
| Régression logistique | 58 % | 28 % | 6 Ko |
| Arbre de décision | 53 % | 48 % | 10 Ko |
| HistGradientBoosting | 69 % | 53 % | 338 Ko |
| **LightGBM (déployé)** | **71 %** | **60 %** | 266 Ko |

Incertitude : ±6 points en validation, ±5 au test. Le boosting bat nettement les modèles simples ; entre variantes
de boosting (y compris des LightGBM plus petits), les écarts restent dans le bruit, et la validation et le test ne
s'accordent pas. LightGBM 150 arbres est donc conservé. Départager plus finement demanderait plus de pannes
(plus de machines simulées).


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
│   ├── main.py             # Routes : prédiction, lecture du parc, info, rechargement à chaud
│   └── schemas.py          # Modèles Pydantic entrée/sortie
├── dashboard.py            # Point d'entrée Streamlit
├── simulation/             # Générateur : python -m simulation.generer
│   ├── meteo.py            # Météo réelle Open-Meteo (ERA5 + CAMS), cache DuckDB
│   ├── physique.py         # Atelier (extérieur --> intérieur) + état caché et risques de panne des machines
│   ├── observation.py      # Défaillances capteurs IoT + écriture DuckDB
│   └── generer.py          # Orchestration
├── src/                    # Pipeline ML : python -m src.train
│   ├── config.py           # Chemins, schéma des données, MLflow
│   ├── donnees.py          # Chargement DuckDB, cible à horizon, split avec tampon et machines réservées
│   ├── features.py         # Features glissantes (une seule fonction pour l'entraînement et l'inférence)
│   ├── evaluation.py       # Métrique événementielle : recall @ budget de fausses alertes
│   ├── modeles.py          # Candidats et comparaison : python -m src.modeles (make compare)
│   ├── train.py            # Modèle retenu + calibration + seuil + promotion @champion
│   ├── predicteur.py       # Chargement du @champion et prédiction
│   ├── explain.py          # Cause probable par famille physique (SHAP)
│   ├── score.py            # Scoring du parc --> table predictions (make score)
│   └── drift.py            # PSI des 30 derniers jours vs entraînement (make check-drift)
├── tests/                  # Suite pytest
├── ui/                     # Modules UI Streamlit
│   ├── constants.py        # Palettes, constantes coûts métier, maps de périodes
│   ├── helpers.py          # Helpers mise en page graphiques, utilitaires de formatage
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
