import pandas as pd
import mlflow
from src.xpredict import Predictor

# Chargement du modèle une fois pour toutes les prédictions
predictor = Predictor(use_mlflow=True)

def process_prediction(machine_data_dict: dict) -> dict:
    """Transforme le dictionnaire validé par Pydantic en DataFrame, exécute la prédiction et formate la sortie.
    """
    df = pd.DataFrame([machine_data_dict])
    results = predictor.predict(df)
    
    prediction_value = int(results["prediction"].iloc[0])
    probability_value = float(results["failure_probability"].iloc[0])
    
    output = {
        "prediction": prediction_value,
        "failure_probability": round(probability_value, 4)
    }

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