import logging
from pathlib import Path
import pandas as pd
import joblib
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Predictor:
    def __init__(self, use_mlflow: bool = True):
        self.use_mlflow = use_mlflow
        self.model_name = "RF_Baseline" 
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
            raise FileNotFoundError(
                f"Aucun modèle trouvé localement à l'adresse {self.local_model_path} ni sur MLflow."
            )

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prend un DataFrame de données brutes, applique le preprocessing et prédit.
        """
        if self.pipeline is None:
            raise ValueError("Le modèle n'est pas chargé.")
        
        logger.info(f"Lancement des prédictions sur {len(data)} lignes...")

        predictions = self.pipeline.predict(data)
        probabilities = self.pipeline.predict_proba(data)[:, 1] # Probabilité de panne
        
        # On construit un DataFrame de résultats propre
        results = data.copy()
        results['prediction'] = predictions
        results['failure_probability'] = probabilities
        
        logger.info("Prédictions terminées avec succès.")
        return results[['prediction', 'failure_probability']]

# Test
if __name__ == "__main__":
    # Jeu de données de test 
    new_data = pd.DataFrame({
       'season': [0, 1, 0],
       'machine_type': [1, 0, 1],
       'ambient_temperature': [25.5, 30.0, 29.8],
       'process_temperature': [35.2, 48.7, 92.3],
       'rotational_speed': [1450, 1480, 2980],
       'torque': [45.6, 47.2, 78.9],
       'tool_wear': [10, 15, 300],
       'machine_load': [0.8, 0.6, 2.9],
       'vibration': [0.1, 0.2, 0.8],
       'humidity': [60, 65, 90],
       'dust': [100, 90, 99],
       'rain_flag': [0, 1, 1],
       'power_outage': [0, 0, 1],
       'voltage': [220, 210, 140],
       'voltage_stability': [1, 0, 1]
    })
    
    # Run de prédiction
    predictor = Predictor(use_mlflow=True)
    predictions = predictor.predict(new_data)
    
    print(predictions)