import logging
import os
from pathlib import Path

import duckdb
import mlflow
import pandas as pd

from src.configloader import load_config
from src.features import enrich_inference_point
from src.xpredict import Predictor

logger = logging.getLogger(__name__)

_config = load_config()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _PROJECT_ROOT / "data" / "raw" / "gdiz_maintenance.db"
_TABLE = "factory_telemetry"

_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", _config["mlflow"]["tracking_uri"])
mlflow.set_tracking_uri(_tracking_uri)

predictor = Predictor(use_mlflow=True)


def _get_machine_history(machine_id: str) -> pd.DataFrame:
    """Charge les 48 dernières lignes d'une machine depuis DuckDB (ordre ASC)."""
    try:
        with duckdb.connect(str(_DB_PATH), read_only=True) as conn:
            df = conn.execute(
                f"""SELECT tool_wear, vibration, process_temperature
                    FROM {_TABLE}
                    WHERE machine_id = ?
                    ORDER BY timestamp DESC LIMIT 48""",
                [machine_id],
            ).fetchdf()
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Impossible de charger l'historique pour {machine_id}: {e}")
        return pd.DataFrame()


def _log_to_mlflow(
    machine_id: str,
    features: dict,
    probability: float,
    prediction: int,
    model_version: str,
) -> None:
    """
    Log une prédiction dans MLflow.
    Expérience dédiée au monitoring — séparée de l'expérience d'entraînement.
    Tags machine_id + alert_level permettent un filtrage direct dans l'UI.
    """
    try:
        mlflow.set_experiment("GDIZ_Monitoring_Predictions")
        alert_level = "high" if probability >= 0.5 else ("medium" if probability >= 0.35 else "normal")
        with mlflow.start_run(
            run_name=machine_id,
            tags={
                "machine_id": machine_id,
                "model_version": model_version,
                "alert_level": alert_level,
            },
        ):
            mlflow.log_metrics({
                "failure_probability": probability,
                "prediction": float(prediction),
                # Métriques capteurs clés
                "tool_wear": float(features.get("tool_wear", 0)),
                "vibration": float(features.get("vibration", 0)),
                "process_temperature": float(features.get("process_temperature", 0)),
                # Features temporelles — signal de dégradation
                "tool_wear_delta_24h": float(features.get("tool_wear_delta_24h", 0)),
                "vibration_max_24h": float(features.get("vibration_max_24h", 0)),
                "process_temp_max_24h": float(features.get("process_temp_max_24h", 0)),
            })
    except Exception as e:
        logger.warning(f"MLflow monitoring log échoué silencieusement : {e}")


def process_prediction(machine_data_dict: dict) -> dict:
    """Valide, enrichit temporellement si machine_id fourni, prédit, log dans MLflow et retourne le résultat."""
    machine_id: str | None = machine_data_dict.pop("machine_id", None)
    features = dict(machine_data_dict)

    if machine_id:
        history_df = _get_machine_history(machine_id)
        features = enrich_inference_point(features, history_df)
    else:
        # Pas d'historique disponible → valeurs neutres (pas d'accélération détectée)
        features["tool_wear_delta_24h"] = 0.0
        features["vibration_max_24h"] = float(features.get("vibration", 0.0))
        features["process_temp_max_24h"] = float(features.get("process_temperature", 0.0))

    df = pd.DataFrame([features])
    results = predictor.predict(df)

    prediction_value = int(results["prediction"].iloc[0])
    probability_value = float(results["failure_probability"].iloc[0])

    _log_to_mlflow(
        machine_id=machine_id or "unknown",
        features=features,
        probability=probability_value,
        prediction=prediction_value,
        model_version=_config["mlflow"]["model_name"],
    )

    return {
        "prediction": prediction_value,
        "failure_probability": round(probability_value, 4),
    }
