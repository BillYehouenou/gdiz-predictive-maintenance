"""Configuration partagée : chemins, tables, cible et registre MLflow."""

import os
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_DB = RACINE / "data" / "raw" / "gdiz_maintenance.db"
TABLE, TABLE_PREDICTIONS = "factory_telemetry", "predictions"
PAS = pd.Timedelta("30min")
JOURS_PREDICTIONS = 60  # seuls les 60 derniers jours sont scorés : dernier risque et courbe de la fiche machine
CIBLE = "cible_h120_predictible"

# Un seul registre : le serveur MLflow de docker-compose (ou de make demo), port 5050 car le 5000 est pris par AirPlay
# sur macOS. La CI le remplace par une base SQLite.
URI_MLFLOW = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
NOM_EXPERIENCE, NOM_MODELE, ALIAS_CHAMPION = "GDIZ_Predictive_Maintenance", "LGBM_Predictive_Maintenance", "champion"
