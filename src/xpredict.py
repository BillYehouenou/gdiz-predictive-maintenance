import json
import logging
import os
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.configloader import charger_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEUIL_DEFAUT = 0.5


class Predicteur:
    def __init__(self, utiliser_mlflow: bool = True, nom_modele: str | None = None):
        self.utiliser_mlflow = utiliser_mlflow
        config = charger_config()
        uri_suivi = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["uri_suivi"])
        mlflow.set_tracking_uri(uri_suivi)
        self.nom_modele = nom_modele or config["mlflow"]["nom_modele"]
        self.chemin_modele_local = PROJECT_ROOT / "models" / "model_pipeline.pkl"
        self.chemin_seuil = PROJECT_ROOT / "models" / "seuil_optimal.json"
        self.pipeline = self._charger_modele()
        self.threshold = self._charger_seuil()

    def _charger_modele(self):
        if self.utiliser_mlflow:
            try:
                model_uri = f"models:/{self.nom_modele}@champion"
                logger.info(f"Chargement depuis MLflow : {model_uri}")
                return mlflow.sklearn.load_model(model_uri)
            except Exception as e:
                logger.warning(f"MLflow indisponible ({e}). Repli sur fichier local.")

        if self.chemin_modele_local.exists():
            logger.info(f"Chargement local : {self.chemin_modele_local}")
            return joblib.load(self.chemin_modele_local)
        raise FileNotFoundError(f"Aucun modèle trouvé : {self.chemin_modele_local} ni MLflow.")

    def _charger_seuil(self) -> float:
        if self.chemin_seuil.exists():
            with open(self.chemin_seuil) as f:
                t = json.load(f)["threshold"]
            logger.info(f"Seuil de décision chargé : {t:.4f}")
            return float(t)
        logger.warning(f"Seuil non trouvé dans {self.chemin_seuil} — utilisation du seuil par défaut 0.5")
        return SEUIL_DEFAUT

    def predire(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Le pipeline chargé produit déjà des probabilités calibrées (calibration
        isotonique ajustée en entraînement sur la validation, cf. src/train.py) —
        pas de recentrage logit ad hoc ici : `probabilite_panne` est directement
        interprétable, `prediction` est la décision au seuil optimal (choisi pour
        respecter le budget de fausses alertes, pas nécessairement proche de 0,5 —
        les pannes restent un événement rare).
        """
        if self.pipeline is None:
            raise ValueError("Le modèle n'est pas chargé.")

        logger.info(f"Prédictions sur {len(data)} ligne(s) avec seuil={self.threshold:.4f}...")

        probabilities: np.ndarray = self.pipeline.predict_proba(data)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        results = pd.DataFrame(
            {
                "prediction": predictions,
                "probabilite_panne": probabilities,
            }
        )

        logger.info("Prédictions terminées.")
        return results
