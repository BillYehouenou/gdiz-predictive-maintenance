import logging
import os
from pathlib import Path

import duckdb
import mlflow
import pandas as pd

from src.configloader import charger_config
from src.features import enrichir_point_inference
from src.xpredict import Predicteur

logger = logging.getLogger(__name__)

_config = charger_config()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _PROJECT_ROOT / "data" / "raw" / "gdiz_maintenance.db"
_TABLE = "factory_telemetry"

_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", _config["mlflow"]["uri_suivi"])
mlflow.set_tracking_uri(_tracking_uri)

predicteur = Predicteur(utiliser_mlflow=True)


def _recuperer_historique_machine(id_machine: str) -> pd.DataFrame:
    """Charge les 48 dernières lignes d'une machine depuis DuckDB (ordre ASC)."""
    try:
        with duckdb.connect(str(_DB_PATH), read_only=True) as conn:
            df = conn.execute(
                f"""SELECT usure_outil, vibration, temperature_process, concentration_poussiere,
                           couple, temperature_ambiante
                    FROM {_TABLE}
                    WHERE id_machine = ?
                    ORDER BY timestamp DESC LIMIT 48""",
                [id_machine],
            ).fetchdf()
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Impossible de charger l'historique pour {id_machine}: {e}")
        return pd.DataFrame()


def _loguer_mlflow(
    id_machine: str,
    features: dict,
    probability: float,
    prediction: int,
    model_version: str,
) -> None:
    """
    Log une prédiction dans MLflow.
    Expérience dédiée au monitoring — séparée de l'expérience d'entraînement.
    Tags id_machine + niveau_alerte permettent un filtrage direct dans l'UI.
    """
    try:
        mlflow.set_experiment("GDIZ_Monitoring_Predictions")
        # Seuils dérivés du seuil de décision réel (probabilités calibrées, cf.
        # docs/model-decisions.md) plutôt que des bornes fixes 0,35/0,50 — un modèle
        # calibré sur un événement rare (~3 % de base) a un seuil optimal typiquement
        # bien plus bas que 0,5.
        threshold = predicteur.threshold
        niveau_alerte = "high" if probability >= threshold * 3 else ("medium" if probability >= threshold else "normal")
        with mlflow.start_run(
            run_name=id_machine,
            tags={
                "id_machine": id_machine,
                "model_version": model_version,
                "niveau_alerte": niveau_alerte,
            },
        ):
            mlflow.log_metrics(
                {
                    "probabilite_panne": probability,
                    "prediction": float(prediction),
                    # Métriques capteurs clés
                    "usure_outil": float(features.get("usure_outil", 0)),
                    "vibration": float(features.get("vibration", 0)),
                    "temperature_process": float(features.get("temperature_process", 0)),
                    # Features temporelles — signal de dégradation
                    "usure_outil_delta_24h": float(features.get("usure_outil_delta_24h", 0)),
                    "vibration_max_24h": float(features.get("vibration_max_24h", 0)),
                    "temperature_process_max_24h": float(features.get("temperature_process_max_24h", 0)),
                }
            )
    except Exception as e:
        logger.warning(f"MLflow monitoring log échoué silencieusement : {e}")


def traiter_prediction(dict_donnees_machine: dict) -> dict:
    """Valide, enrichit temporellement si id_machine fourni, prédit, log dans MLflow et retourne le résultat."""
    id_machine: str | None = dict_donnees_machine.pop("id_machine", None)
    features = dict(dict_donnees_machine)

    if id_machine:
        history_df = _recuperer_historique_machine(id_machine)
        features = enrichir_point_inference(features, history_df)
    else:
        # Pas d'historique disponible - valeurs neutres (pas d'accélération détectée)
        vibr = float(features.get("vibration", 0.0))
        temp = float(features.get("temperature_process", 0.0))
        ambient = float(features.get("temperature_ambiante", 0.0))
        dust = float(features.get("concentration_poussiere", 0.0))
        couple = float(features.get("couple", 0.0))
        features["usure_outil_delta_24h"] = 0.0
        features["usure_outil_delta_4h"] = 0.0
        features["vibration_max_24h"] = vibr
        features["vibration_moyenne_24h"] = vibr
        features["vibration_ecart_type_24h"] = 0.0
        features["vibration_delta_4h"] = 0.0
        features["temperature_process_max_24h"] = temp
        features["temperature_process_moyenne_24h"] = temp
        features["temperature_process_delta_4h"] = 0.0
        features["ecart_thermique"] = temp - ambient
        features["ecart_thermique_max_24h"] = temp - ambient
        features["poussiere_moyenne_24h"] = dust
        features["poussiere_max_24h"] = dust
        features["couple_max_24h"] = couple

    df = pd.DataFrame([features])
    results = predicteur.predire(df)

    valeur_prediction = int(results["prediction"].iloc[0])
    valeur_probabilite = float(results["probabilite_panne"].iloc[0])

    _loguer_mlflow(
        id_machine=id_machine or "unknown",
        features=features,
        probability=valeur_probabilite,
        prediction=valeur_prediction,
        model_version=_config["mlflow"]["nom_modele"],
    )

    return {
        "prediction": valeur_prediction,
        "probabilite_panne": round(valeur_probabilite, 4),
    }
