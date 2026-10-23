"""
Drift : compare les derniers jours de télémétrie et de risque prédit à la période d'entraînement.
Un drift est un signal à creuser (capteur, saison, process), pas une alerte automatique.
Usage : python -m src.drift
"""

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import CHEMIN_DB, TABLE_PREDICTIONS
from src.donnees import DATE_TEST, charger

VARIABLES = [
    "heures_depuis_changement_outil",
    "vibration",
    "temperature_process",
    "concentration_poussiere",
    "vibration_max_24h",
    "temperature_process_max_24h",
    "probabilite_panne",
]
# Avec des centaines de milliers de lignes, la p-value du KS vaut toujours ~0 : on décide sur le PSI
SEUIL_PSI = 0.25  # repère usuel : < 0,1 stable, 0,1-0,25 modéré, > 0,25 significatif


def psi(reference: pd.Series, production: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index sur les déciles de la référence."""
    reference, production = reference.dropna(), production.dropna()
    if reference.empty or production.empty or reference.nunique() < 2:
        return 0.0
    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)))
    if len(bornes) < 3:
        return 0.0
    bornes[0], bornes[-1] = -np.inf, np.inf
    part_ref = (pd.cut(reference, bornes).value_counts(sort=False) / len(reference)).clip(lower=1e-6)
    part_prod = (pd.cut(production, bornes).value_counts(sort=False) / len(production)).clip(lower=1e-6)
    return float(np.sum((part_prod - part_ref) * np.log(part_prod / part_ref)))


def verifier_derive(df: pd.DataFrame, jours: int = 30) -> pd.DataFrame:
    """df : télémétrie avec features et probabilite_panne. Référence = avant le test, production = derniers jours."""
    reference = df[df["timestamp"] < DATE_TEST]
    production = df[df["timestamp"] > df["timestamp"].max() - pd.Timedelta(days=jours)]
    lignes = []
    for variable in VARIABLES:
        ref, prod = reference[variable].dropna(), production[variable].dropna()
        if ref.empty or prod.empty:
            continue
        indice = psi(ref, prod)
        lignes.append(
            {
                "variable": variable,
                "ks": round(float(ks_2samp(ref, prod).statistic), 3),
                "psi": round(indice, 3),
                "drift": indice > SEUIL_PSI,
            }
        )
    return pd.DataFrame(lignes)


def main():
    with duckdb.connect(str(CHEMIN_DB), read_only=True) as conn:
        predictions = conn.execute(f"SELECT id_machine, timestamp, probabilite_panne FROM {TABLE_PREDICTIONS}").fetchdf()
    df = charger().merge(predictions, on=["id_machine", "timestamp"], how="left")
    print(verifier_derive(df).to_string(index=False))


if __name__ == "__main__":
    main()
