# Maintenance Prédictive Industrielle GDIZ

Plateforme MLOps de bout en bout pour la prédiction de pannes en temps réel sur des machines textiles de la Zone Industrielle de Glo-Djigbé, Bénin.


## Résumé

Cette plateforme opérationnalise un modèle LightGBM de prédiction de pannes entraîné sur 1,75 million de relevés IoT (50 machines × 730 jours × 48 pas de 30 minutes). Elle expose une intelligence de maintenance actionnable via un tableau de bord web (HTML, CSS et JavaScript servis par FastAPI) adossé à un moteur analytique DuckDB embarqué, avec MLflow gérant l'intégralité du cycle de vie des expériences. Le système est conçu pour les contraintes opérationnelles spécifiques du parc textile : déséquilibre de classes extrême, instabilité du réseau SBEE et cycles de poussière Harmattan.

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
        │                              identique à l'entraînement, au scoring et dans l'API
        │                              split train / validation / test avec tampon de 120h + 10 machines réservées au test
        │
src/train.py                         - LightGBM (NaN et catégories natives) --> calibration sigmoïde --> seuil choisi
        │                              sur la validation (recall événementiel sous budget de fausses alertes)
        │                              --> MLflow Registry : @champion remplacé seulement si le recall ne régresse pas
        │
src/score.py                         - score tout le parc une fois --> table DuckDB `predictions`
        │                              (risque, niveau, cause probable SHAP des machines en alerte)
        │
app/main.py (FastAPI)                - API et tableau de bord sur le même port
  GET  /  · /machines/{id}                           pages web (web/templates + web/static)
  GET  /api/v1/parc?periode= · /etat · /machines/{id} · /machines/{id}/risque?jours=
  POST /api/v1/predict(/batch) · GET /api/v1/info · POST /api/v1/reload
```


## Tableau de bord

Pensé pour le superviseur ou le chef d'usine, en deux écrans :

- **Parc** : disponibilité, MTBF, coût réel des pannes et heures d'arrêt sur la période choisie, chacun
  expliqué en termes métier (« Voir le détail ») ; plan des
  50 machines par état ; matrice risque × criticité (criticité = part de la production perdue si la machine
  s'arrête, déduite du nombre de machines à son étape de la chaîne).
- **Machine** : santé, risque à 5 jours, contributions SHAP, frise des pannes, capteurs sur 30 jours et
  courbe de risque brute (pas de 2 h, délestages visibles).

Le HTML ne contient aucune donnée : chaque écran lit l'API, avec un état de chargement, vide et d'erreur.
« Exporter le rapport » produit un PDF de l'écran. Thème clair ou sombre, utilisable au clavier et sur téléphone.


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

# API et tableau de bord sur http://localhost:8000 (documentation de l'API : /docs)
make serve

# Optionnel : MLflow UI
make mlflow-ui
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

La stack Compose orchestre deux services : l'API (avec le tableau de bord, port 8000) et MLflow (port 5000).

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
│   ├── main.py             # Pages web, routes des deux écrans, prédiction
│   └── schemas.py          # Modèles Pydantic entrée/sortie
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
│   ├── usine.py            # Chaîne de production, criticité, coûts par gamme
│   ├── metier.py           # Indicateurs, coût réel des pannes, fiche machine (lecture DuckDB)
│   ├── score.py            # Scoring du parc --> table predictions (make score)
│   └── drift.py            # PSI des 30 derniers jours vs entraînement (make check-drift)
├── tests/                  # Suite pytest
├── web/                    # Tableau de bord (sans framework ni étape de build)
│   ├── templates/          # Gabarit commun + Parc et Machine (Jinja2)
│   └── static/
│       ├── css/            # tokens.css (couleurs, thèmes clair/sombre) + app.css
│       └── js/             # api.js, nav.js, charts.js (SVG) + un script par écran
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```
