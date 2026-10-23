"""Chargement du modèle @champion (MLflow, ou fichier local en repli) et prédiction."""

import json
import logging

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import DOSSIER_MODELES, NOM_MODELE, URI_MLFLOW
from src.features import preparer_X

logger = logging.getLogger(__name__)

SEUIL_DEFAUT = 0.5


class Predicteur:
    def __init__(self, utiliser_mlflow: bool = True):
        mlflow.set_tracking_uri(URI_MLFLOW)
        self.nom_modele = NOM_MODELE
        self.version = "local"
        self.modele = self._charger_modele(utiliser_mlflow)
        chemin_seuil = DOSSIER_MODELES / "seuil_optimal.json"
        self.threshold = json.loads(chemin_seuil.read_text())["threshold"] if chemin_seuil.exists() else SEUIL_DEFAUT

    def _charger_modele(self, utiliser_mlflow: bool):
        if utiliser_mlflow:
            try:
                modele = mlflow.sklearn.load_model(f"models:/{NOM_MODELE}@champion")
                self.version = str(mlflow.MlflowClient().get_model_version_by_alias(NOM_MODELE, "champion").version)
                return modele
            except Exception as e:
                logger.warning(f"MLflow indisponible ({e}), repli sur le fichier local.")

        chemin = DOSSIER_MODELES / "model_pipeline.pkl"
        if not chemin.exists():
            raise FileNotFoundError(f"Aucun modèle : ni @champion MLflow, ni {chemin}.")
        return joblib.load(chemin)

    def predire(self, df: pd.DataFrame) -> pd.DataFrame:
        """df : lignes avec features calculées (src.features.calculer_features)."""
        probas = self.modele.predict_proba(preparer_X(df))[:, 1]
        return pd.DataFrame({"prediction": (probas >= self.threshold).astype(int), "probabilite_panne": probas})
