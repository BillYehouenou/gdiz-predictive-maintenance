"""
Détection de drift — uv run python scripts/check_drift.py

Compare la distribution des features reçues en production (loggées dans MLflow
via app/utils.py::_log_to_mlflow, expérience GDIZ_Monitoring_Predictions) à celle
du jeu d'entraînement, via un test de Kolmogorov-Smirnov par feature.

Une p-value faible (< 0.05) signifie que les deux distributions sont statistiquement
différentes — signal qu'il faut creuser (capteur défaillant, changement de process,
nouvelle saison non représentée à l'entraînement...), pas une alerte automatique de
mauvaise qualité du modèle en soi.
"""

import logging

import mlflow
import pandas as pd
from scipy.stats import ks_2samp

from src.dataloader import DataLoader
from src.train import SPLIT_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "GDIZ_Monitoring_Predictions"
DRIFT_FEATURES = [
    "tool_wear",
    "vibration",
    "process_temperature",
    "tool_wear_delta_24h",
    "vibration_max_24h",
    "process_temp_max_24h",
]
SIGNIFICANCE_LEVEL = 0.05


def _load_reference_distribution() -> pd.DataFrame:
    """Sous-ensemble d'entraînement (avant SPLIT_DATE) — référence vue par le modèle."""
    df = DataLoader().load_raw_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[df["timestamp"] < SPLIT_DATE][DRIFT_FEATURES]


def _load_production_distribution() -> pd.DataFrame:
    """Features des prédictions récentes, loggées comme métriques MLflow."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame(columns=DRIFT_FEATURES)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=5000)
    cols = {f: f"metrics.{f}" for f in DRIFT_FEATURES if f"metrics.{f}" in runs.columns}
    return runs[list(cols.values())].rename(columns={v: k for k, v in cols.items()})


def check_drift() -> list[dict]:
    reference = _load_reference_distribution()
    production = _load_production_distribution()

    if production.empty:
        logger.warning("Aucune prédiction loggée dans MLflow — rien à comparer.")
        return []

    results = []
    for feature in DRIFT_FEATURES:
        if feature not in production.columns or production[feature].dropna().empty:
            continue
        statistic, p_value = ks_2samp(reference[feature].dropna(), production[feature].dropna())
        results.append(
            {
                "feature": feature,
                "ks_statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 4),
                "drift_detecte": p_value < SIGNIFICANCE_LEVEL,
            }
        )
    return results


if __name__ == "__main__":
    report = check_drift()
    if not report:
        print("Pas assez de données pour évaluer le drift.")
    else:
        print(f"\n{'Feature':<25}{'KS':>10}{'p-value':>12}{'Drift':>10}")
        print("-" * 57)
        for row in report:
            flag = "OUI ⚠️" if row["drift_detecte"] else "non"
            print(f"{row['feature']:<25}{row['ks_statistic']:>10}{row['p_value']:>12}{flag:>10}")
