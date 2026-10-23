"""
Détection de drift — uv run python scripts/check_drift.py

Compare la distribution des features reçues en production à celle du jeu
d'entraînement, et la distribution des probabilités prédites elles-mêmes (pas
seulement les features en entrée) — deux tests complémentaires par variable
(Kolmogorov-Smirnov + PSI), et un troisième sur la sortie du modèle.

Source des données de production : l'expérience MLflow GDIZ_Monitoring_Predictions
(loggée par app/utils.py::_loguer_mlflow, une prédiction = un run). C'est la « table
de prédictions dédiée » du projet — interrogeable via `mlflow.search_runs`, pas
besoin d'une deuxième table DuckDB qui dupliquerait la même donnée.

Une p-value KS faible (< 0.05) ou un PSI élevé (> 0.25, repère usuel de l'industrie)
signifient que les deux distributions sont statistiquement différentes — signal
qu'il faut creuser (capteur défaillant, changement de process, nouvelle saison non
représentée à l'entraînement, dérive du comportement du modèle...), pas une alerte
automatique de mauvaise qualité en soi.
"""

import logging

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.configloader import charger_config
from src.dataloader import ChargeurDonnees
from src.train import DATE_COUPURE
from src.xpredict import Predicteur

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOM_EXPERIENCE = "GDIZ_Monitoring_Predictions"
VARIABLES_DERIVE = [
    "heures_depuis_changement_outil",
    "vibration",
    "temperature_process",
    "concentration_poussiere",
    "vibration_max_24h",
    "temperature_process_max_24h",
]
SEUIL_SIGNIFICATIVITE = 0.05
SEUIL_PSI = 0.25  # repère usuel : <0.1 stable, 0.1-0.25 dérive modérée, >0.25 significative
TAILLE_ECHANTILLON_REFERENCE = 20_000  # sous-échantillon pour garder le check rapide


def _psi(reference: pd.Series, production: pd.Series, buckets: int = 10) -> float:
    """
    Population Stability Index : bucket la référence en quantiles, compare la part
    de la production tombant dans chaque bucket à la part attendue (référence).
    Contrairement au KS (sensible à la taille d'échantillon), le PSI reste stable
    sur de petits volumes de production — les deux tests se complètent.
    """
    reference = reference.dropna()
    production = production.dropna()
    if reference.empty or production.empty or reference.nunique() < 2:
        return 0.0

    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf

    eps = 1e-6
    part_reference = pd.cut(reference, quantiles).value_counts(sort=False) / len(reference)
    part_production = pd.cut(production, quantiles).value_counts(sort=False) / len(production)
    part_reference = part_reference.clip(lower=eps)
    part_production = part_production.clip(lower=eps)
    return float(np.sum((part_production - part_reference) * np.log(part_production / part_reference)))


def _charger_distribution_reference() -> pd.DataFrame:
    """Sous-ensemble d'entraînement (avant DATE_COUPURE) — référence vue par le modèle."""
    df = ChargeurDonnees().charger_donnees_brutes()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[df["timestamp"] < DATE_COUPURE]


def _charger_distribution_production(variables: list[str]) -> pd.DataFrame:
    """Métriques des prédictions récentes, loggées dans l'expérience MLflow de monitoring."""
    experience = mlflow.get_experiment_by_name(NOM_EXPERIENCE)
    if experience is None:
        return pd.DataFrame(columns=variables)
    runs = mlflow.search_runs(experiment_ids=[experience.experiment_id], max_results=5000)
    colonnes = {v: f"metrics.{v}" for v in variables if f"metrics.{v}" in runs.columns}
    return runs[list(colonnes.values())].rename(columns={v: k for k, v in colonnes.items()})


def verifier_derive() -> list[dict]:
    """Drift feature par feature — signal d'entrée du modèle."""
    reference = _charger_distribution_reference()[VARIABLES_DERIVE]
    production = _charger_distribution_production(VARIABLES_DERIVE)

    if production.empty:
        logger.warning("Aucune prédiction loggée dans MLflow — rien à comparer.")
        return []

    resultats = []
    for variable in VARIABLES_DERIVE:
        if variable not in production.columns or production[variable].dropna().empty:
            continue
        ref_col, prod_col = reference[variable].dropna(), production[variable].dropna()
        statistique, p_value = ks_2samp(ref_col, prod_col)
        psi = _psi(ref_col, prod_col)
        resultats.append(
            {
                "variable": variable,
                "statistique_ks": round(float(statistique), 4),
                "p_value": round(float(p_value), 4),
                "psi": round(psi, 4),
                "drift_detecte": p_value < SEUIL_SIGNIFICATIVITE or psi > SEUIL_PSI,
            }
        )
    return resultats


def verifier_derive_predictions() -> dict | None:
    """
    Drift de la sortie du modèle elle-même : la distribution de probabilite_panne
    prédite en production a-t-elle bougé par rapport à ce que le modèle produisait
    sur la période de référence ? Peut se déclencher même quand les features en
    entrée semblent stables (interaction de features, dérive de calibration lente).
    """
    production = _charger_distribution_production(["probabilite_panne", "prediction"])
    if production.empty or production["probabilite_panne"].dropna().empty:
        logger.warning("Aucune probabilité loggée dans MLflow — drift des prédictions non évalué.")
        return None

    config = charger_config()
    colonnes_features = config["jeu_donnees"]["features"]
    reference_df = _charger_distribution_reference()
    if len(reference_df) > TAILLE_ECHANTILLON_REFERENCE:
        reference_df = reference_df.sample(n=TAILLE_ECHANTILLON_REFERENCE, random_state=42)

    predicteur = Predicteur(utiliser_mlflow=True)
    probas_reference = predicteur.predire(reference_df[colonnes_features])["probabilite_panne"]
    probas_production = production["probabilite_panne"].dropna()

    statistique, p_value = ks_2samp(probas_reference, probas_production)
    psi = _psi(probas_reference, probas_production)
    return {
        "taux_alerte_reference": round(float((probas_reference >= predicteur.threshold).mean()), 4),
        "taux_alerte_production": round(float(production["prediction"].dropna().mean()), 4),
        "statistique_ks": round(float(statistique), 4),
        "p_value": round(float(p_value), 4),
        "psi": round(psi, 4),
        "drift_detecte": p_value < SEUIL_SIGNIFICATIVITE or psi > SEUIL_PSI,
    }


if __name__ == "__main__":
    rapport = verifier_derive()
    if not rapport:
        print("Pas assez de données pour évaluer le drift des features.")
    else:
        print(f"\n{'Variable':<25}{'KS':>10}{'p-value':>12}{'PSI':>10}{'Drift':>10}")
        print("-" * 67)
        for ligne in rapport:
            drapeau = "OUI ⚠️" if ligne["drift_detecte"] else "non"
            print(f"{ligne['variable']:<25}{ligne['statistique_ks']:>10}{ligne['p_value']:>12}{ligne['psi']:>10}{drapeau:>10}")

    rapport_predictions = verifier_derive_predictions()
    print("\n--- Drift des prédictions (sortie du modèle) ---")
    if rapport_predictions is None:
        print("Pas assez de données pour évaluer le drift des prédictions.")
    else:
        drapeau = "OUI ⚠️" if rapport_predictions["drift_detecte"] else "non"
        print(
            f"Taux d'alerte référence={rapport_predictions['taux_alerte_reference']:.1%} vs "
            f"production={rapport_predictions['taux_alerte_production']:.1%} | "
            f"KS={rapport_predictions['statistique_ks']} p={rapport_predictions['p_value']} "
            f"PSI={rapport_predictions['psi']} | Drift={drapeau}"
        )
