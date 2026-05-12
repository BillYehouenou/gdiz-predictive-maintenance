import pandas as pd
from src.xpredict import Predictor

# Chargement du modèle une fois pour toutes les prédictions
predictor = Predictor(use_mlflow=True)

def process_prediction(machine_data_dict: dict) -> dict:
    """Transforme le dictionnaire validé par Pydantic en DataFrame, exécute la prédiction et formate la sortie.
    """
    df = pd.DataFrame([machine_data_dict])
    results = predictor.predict(df)
    
    return {
        "prediction": int(results["prediction"].iloc[0]),
        "failure_probability": float(results["failure_probability"].iloc[0])
    }