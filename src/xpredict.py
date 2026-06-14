import logging
import os
from pathlib import Path
import joblib
import mlflow.sklearn
import pandas as pd
from src.configloader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Predictor:
    def __init__(self, use_mlflow: bool = True, model_name: str = None):
        self.use_mlflow = use_mlflow
        config = load_config()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        self.model_name = model_name or config["mlflow"]["model_name"]
        self.local_model_path = PROJECT_ROOT / "models" / "model_pipeline.pkl"
        self.pipeline = self._load_model()

    def _load_model(self):
        """
        Charge le pipeline de manière sécurisée (depuis MLflow ou en local).
        """
        if self.use_mlflow:
            try:
                # On va chercher la version la plus récente du modèle enregistré
                model_uri = f"models:/{self.model_name}/latest"
                logger.info(f"Chargement du modèle de production depuis MLflow : {model_uri}")
                return mlflow.sklearn.load_model(model_uri)
            except Exception as e:
                logger.warning(f"Impossible de charger depuis MLflow ({e}). Repli sur le fichier local.")

        # Repli ou chargement local
        if self.local_model_path.exists():
            logger.info(f"Chargement du modèle local sécurisé : {self.local_model_path}")
            return joblib.load(self.local_model_path)
        else:
            raise FileNotFoundError(f"Aucun modèle trouvé localement à l'adresse {self.local_model_path} ni sur MLflow.")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prend un DataFrame de données brutes, applique le preprocessing et prédit.
        """
        if self.pipeline is None:
            raise ValueError("Le modèle n'est pas chargé.")

        logger.info(f"Lancement des prédictions sur {len(data)} lignes...")

        predictions = self.pipeline.predict(data)
        probabilities = self.pipeline.predict_proba(data)[:, 1]

        results = data.copy()
        results["prediction"] = predictions
        results["failure_probability"] = probabilities

        logger.info("Prédictions terminées avec succès.")
        return results[["prediction", "failure_probability"]]


if __name__ == "__main__":
    new_data = pd.DataFrame(
        {
            "benin_season": [1, 0, 1],
            "machine_type": ["H", "M", "L"],
            "ambient_temperature": [25.5, 30.0, 41.0],
            "process_temperature": [38.2, 52.7, 89.0],
            "rotational_speed": [1200, 1450, 2500],
            "torque": [30.0, 42.0, 85.0],
            "tool_wear": [15.0, 40.0, 90.0],
            "activity_level": [0.5, 0.7, 0.95],
            "vibration": [1.2, 2.0, 7.5],
            "humidity": [65, 50, 25],
            "dust_concentration": [80, 150, 380],
            "rain_flag": [1, 1, 0],
            "power_loss_indicator": [0, 0, 1],
            "voltage_level": [225.0, 218.0, 195.0],
            "voltage_stability": [80.0, 65.0, 20.0],
        }
    )

    predictor = Predictor(use_mlflow=True)
    predictions = predictor.predict(new_data)
    print(predictions)
