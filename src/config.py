"""Configuration partagée : chemins, schéma des données, MLflow."""

import os
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_DB = RACINE / "data" / "raw" / "gdiz_maintenance.db"
TABLE = "factory_telemetry"
TABLE_PREDICTIONS = "predictions"
JOURS_PREDICTIONS = 60  # seuls les 60 derniers jours sont scorés : dernier risque et courbe de la fiche machine
PAS = pd.Timedelta("30min")

# Mesures brutes vues par le modèle : capteurs + compteurs GMAO (les features glissantes sont dans src/features.py)
FEATURES_BRUTES = [
    "type_machine",
    "temperature_ambiante",
    "temperature_process",
    "vitesse_rotation",
    "couple",
    "niveau_activite",
    "vibration",
    "humidite",
    "concentration_poussiere",
    "precipitation_mm",
    "indicateur_coupure",
    "niveau_tension",
    "heures_depuis_changement_outil",
    "heures_depuis_maintenance",
]
CIBLE = "cible_h120_predictible"

# Un seul registre pour tout : le serveur MLflow de docker-compose (make mlflow), utilisé par l'entraînement,
# le scoring et l'API. Port 5050 : sur macOS, le 5000 est pris par AirPlay. La CI le remplace par SQLite.
URI_MLFLOW = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
NOM_EXPERIENCE = "GDIZ_Predictive_Maintenance"
NOM_MODELE = "LGBM_Predictive_Maintenance"
ALIAS_CHAMPION = "champion"
