import json
import logging
import os
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.configloader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLD = 0.5


class Predictor:
    def __init__(self, use_mlflow: bool = True, model_name: str | None = None):
        self.use_mlflow = use_mlflow
        config = load_config()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        self.model_name = model_name or config["mlflow"]["model_name"]
        self.local_model_path = PROJECT_ROOT / "models" / "model_pipeline.pkl"
        self.threshold_path = PROJECT_ROOT / "models" / "optimal_threshold.json"
        self.pipeline = self._load_model()
        self.threshold = self._load_threshold()

    def _load_model(self):
        if self.use_mlflow:
            try:
                model_uri = f"models:/{self.model_name}@champion"
                logger.info(f"Chargement depuis MLflow : {model_uri}")
                return mlflow.sklearn.load_model(model_uri)
            except Exception as e:
                logger.warning(f"MLflow indisponible ({e}). Repli sur fichier local.")

        if self.local_model_path.exists():
            logger.info(f"Chargement local : {self.local_model_path}")
            return joblib.load(self.local_model_path)
        raise FileNotFoundError(f"Aucun modèle trouvé : {self.local_model_path} ni MLflow.")

    def _load_threshold(self) -> float:
        if self.threshold_path.exists():
            with open(self.threshold_path) as f:
                t = json.load(f)["threshold"]
            logger.info(f"Seuil de décision chargé : {t:.4f}")
            return float(t)
        logger.warning(f"Seuil non trouvé dans {self.threshold_path} — utilisation du seuil par défaut 0.5")
        return DEFAULT_THRESHOLD

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Le pipeline chargé produit déjà des probabilités calibrées (calibration
        isotonique ajustée en entraînement sur la validation, cf. src/train.py) —
        pas de recentrage logit ad hoc ici : `failure_probability` est directement
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
                "failure_probability": probabilities,
            }
        )

        logger.info("Prédictions terminées.")
        return results
