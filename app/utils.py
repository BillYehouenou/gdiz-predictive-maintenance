import os

import mlflow
import pandas as pd

from src.configloader import load_config
from src.xpredict import Predictor

_config = load_config()
_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", _config["mlflow"]["tracking_uri"])
mlflow.set_tracking_uri(_tracking_uri)

predictor = Predictor(use_mlflow=True)


def process_prediction(machine_data_dict: dict) -> dict:
    """Transforme le dictionnaire validé par Pydantic en DataFrame, exécute la prédiction et formate la sortie."""
    df = pd.DataFrame([machine_data_dict])
    results = predictor.predict(df)

    prediction_value = int(results["prediction"].iloc[0])
    probability_value = float(results["failure_probability"].iloc[0])

    output = {"prediction": prediction_value, "failure_probability": round(probability_value, 4)}

    # Monitoring : Enregistrement dans MLflow
    # On utilise un try/except pour que l'API ne crash pas si MLflow a un souci
    try:
        mlflow.set_experiment("GDIZ_Monitoring_API")

        with mlflow.start_run(run_name="prediction_log"):
            # On log les entrées (température, vitesse, etc.)
            mlflow.log_params(machine_data_dict)
            # On log les résultats
            mlflow.log_metric("pred_class", prediction_value)
            mlflow.log_metric("pred_prob", probability_value)

    except Exception as e:
        print(f"Erreur de monitoring MLflow : {e}")

    return output
