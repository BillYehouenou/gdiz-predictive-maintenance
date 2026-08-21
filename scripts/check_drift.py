"""
Détection de drift — uv run python scripts/check_drift.py

Compare la distribution des features reçues en production (loggées dans MLflow
via app/utils.py::_loguer_mlflow, expérience GDIZ_Monitoring_Predictions) à celle
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

from src.dataloader import ChargeurDonnees
from src.train import DATE_COUPURE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOM_EXPERIENCE = "GDIZ_Monitoring_Predictions"
VARIABLES_DERIVE = [
    "usure_outil",
    "vibration",
    "temperature_process",
    "usure_outil_delta_24h",
    "vibration_max_24h",
    "temperature_process_max_24h",
]
SEUIL_SIGNIFICATIVITE = 0.05


def _charger_distribution_reference() -> pd.DataFrame:
    """Sous-ensemble d'entraînement (avant DATE_COUPURE) — référence vue par le modèle."""
    df = ChargeurDonnees().charger_donnees_brutes()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[df["timestamp"] < DATE_COUPURE][VARIABLES_DERIVE]


def _charger_distribution_production() -> pd.DataFrame:
    """Features des prédictions récentes, loggées comme métriques MLflow."""
    experience = mlflow.get_experiment_by_name(NOM_EXPERIENCE)
    if experience is None:
        return pd.DataFrame(columns=VARIABLES_DERIVE)
    runs = mlflow.search_runs(experiment_ids=[experience.experiment_id], max_results=5000)
    colonnes = {v: f"metrics.{v}" for v in VARIABLES_DERIVE if f"metrics.{v}" in runs.columns}
    return runs[list(colonnes.values())].rename(columns={v: k for k, v in colonnes.items()})


def verifier_derive() -> list[dict]:
    reference = _charger_distribution_reference()
    production = _charger_distribution_production()

    if production.empty:
        logger.warning("Aucune prédiction loggée dans MLflow — rien à comparer.")
        return []

    resultats = []
    for variable in VARIABLES_DERIVE:
        if variable not in production.columns or production[variable].dropna().empty:
            continue
        statistique, p_value = ks_2samp(reference[variable].dropna(), production[variable].dropna())
        resultats.append(
            {
                "variable": variable,
                "statistique_ks": round(float(statistique), 4),
                "p_value": round(float(p_value), 4),
                "drift_detecte": p_value < SEUIL_SIGNIFICATIVITE,
            }
        )
    return resultats


if __name__ == "__main__":
    rapport = verifier_derive()
    if not rapport:
        print("Pas assez de données pour évaluer le drift.")
    else:
        print(f"\n{'Variable':<25}{'KS':>10}{'p-value':>12}{'Drift':>10}")
        print("-" * 57)
        for ligne in rapport:
            drapeau = "OUI ⚠️" if ligne["drift_detecte"] else "non"
            print(f"{ligne['variable']:<25}{ligne['statistique_ks']:>10}{ligne['p_value']:>12}{drapeau:>10}")
