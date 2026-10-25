"""
Suivi du modèle en service : python -m src.drift (make check-drift).
- performance réelle des 60 derniers jours : les pannes passées sont connues, on les compare aux alertes qui précédaient ;
- dérive des capteurs : les 30 derniers jours face aux mêmes jours un an plus tôt (la météo est saisonnière).
Une dérive est un signal à creuser, pas une alerte automatique.
"""

import duckdb
import numpy as np
import pandas as pd

from src.config import CHEMIN_DB, JOURS_PREDICTIONS, TABLE_PREDICTIONS
from src.donnees import charger
from src.evaluation import PAS_HORIZON, bilan_economique, metriques_evenementielles

# Mesures des machines ; les compteurs (heures d'outil, de maintenance) grandissent avec le temps et dériveraient toujours
VARIABLES = ["vibration", "temperature_process", "concentration_poussiere", "vibration_max_24h", "temperature_process_max_24h"]
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
    """df : télémétrie avec features. Production = derniers jours ; référence = la même période un an plus tôt."""
    fin = df["timestamp"].max()
    il_y_a_un_an = fin - pd.DateOffset(years=1)
    production = df[df["timestamp"] > fin - pd.Timedelta(days=jours)]
    reference = df[(df["timestamp"] > il_y_a_un_an - pd.Timedelta(days=jours)) & (df["timestamp"] <= il_y_a_un_an)]
    lignes = [
        {"variable": v, "psi": round(indice := psi(reference[v], production[v]), 3), "drift": indice > SEUIL_PSI}
        for v in VARIABLES
        if reference[v].notna().any() and production[v].notna().any()
    ]
    return pd.DataFrame(lignes)


def performance_recente(df: pd.DataFrame, seuil: float, jours: int = JOURS_PREDICTIONS) -> dict:
    """df : télémétrie avec probabilite_panne (vide hors marche). Mêmes mesures qu'au test, sur la période récente."""
    recent = df[df["timestamp"] > df["timestamp"].max() - pd.Timedelta(days=jours)].reset_index(drop=True)
    metriques = metriques_evenementielles(recent, recent["probabilite_panne"].fillna(0.0).to_numpy(), seuil, PAS_HORIZON)
    return metriques | bilan_economique(recent, metriques)


def main():
    from src.modele import Predicteur  # le seuil vient du @champion, comme pour l'API

    seuil = Predicteur().threshold
    with duckdb.connect(str(CHEMIN_DB), read_only=True) as conn:
        predictions = conn.execute(f"SELECT id_machine, timestamp, probabilite_panne FROM {TABLE_PREDICTIONS}").fetchdf()
    df = charger().merge(predictions, on=["id_machine", "timestamp"], how="left")
    p = performance_recente(df, seuil)
    utiles = round(p["recall_preavis_4h"] * p["nb_pannes"])
    print(f"Performance des {JOURS_PREDICTIONS} derniers jours (seuil {seuil:.2%}) :")
    print(f"  {p['nb_attrapees']} pannes prévisibles annoncées sur {p['nb_pannes']}, dont {utiles} avec au moins 4 h de préavis")
    print(
        f"  précision des alertes {p['precision_alertes']:.1%} · {p['taux_fausses_alertes_par_machine_jour']:.2f} fausse alerte"
    )
    print(f"  par machine et par jour · gain net {p['gain_net'] / 1e6:.1f} M FCFA (peu de pannes : à lire comme une tendance)")
    print("\nDérive des capteurs, 30 derniers jours face aux mêmes jours un an plus tôt :")
    print(verifier_derive(df).to_string(index=False))


if __name__ == "__main__":
    main()
