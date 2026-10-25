"""Configuration partagée : chemins, schéma des données, MLflow."""

import os
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_DB = RACINE / "data" / "raw" / "gdiz_maintenance.db"
DOSSIER_MODELES = RACINE / "models"
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

URI_MLFLOW = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
NOM_EXPERIENCE = "GDIZ_Predictive_Maintenance"
NOM_MODELE = "LGBM_Predictive_Maintenance"
