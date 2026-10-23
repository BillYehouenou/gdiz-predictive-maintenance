"""Chargement de la télémétrie, cible à horizon et découpage train / validation / test."""

import logging
import pathlib

import duckdb
import numpy as np
import pandas as pd

from src.config import CHEMIN_DB, CIBLE, TABLE
from src.features import TYPES_MACHINE, calculer_features

logger = logging.getLogger(__name__)

MODES_PANNE_PREDICTIBLES = ("TWF", "HDF", "OSF")  # PWF et RNF n'ont pas de signe avant-coureur
HORIZON = pd.Timedelta("120h")
DATE_VALIDATION = pd.Timestamp("2024-10-01")
DATE_TEST = pd.Timestamp("2025-01-01")
# Fenêtres de validation pour comparer les modèles : chaque trimestre 2024 après le premier
PLIS = [
    (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-07-01")),
    (pd.Timestamp("2024-07-01"), DATE_VALIDATION),
    (DATE_VALIDATION, DATE_TEST),
]
# Une machine sur cinq n'est vue qu'au test : mesure la généralisation à une machine neuve
MACHINES_RESERVEES = [f"GDIZ_TEX_M_{i:02d}" for i in range(5, 51, 5)]

# Bornes physiques larges : une valeur hors bornes est signalée, jamais bloquante
BORNES = {
    "temperature_process": (-50, 200),
    "temperature_ambiante": (-20, 60),
    "vitesse_rotation": (-1000, 5000),
    "couple": (-100, 500),
    "vibration": (-10, 50),
    "humidite": (-20, 150),
    "concentration_poussiere": (-100, 2000),
    "precipitation_mm": (0, 200),
    "niveau_tension": (0, 400),
    "niveau_activite": (-0.5, 1.5),
    "heures_depuis_changement_outil": (0, 50_000),
    "heures_depuis_maintenance": (0, 50_000),
}


def charger(chemin: pathlib.Path = CHEMIN_DB) -> pd.DataFrame:
    """Télémétrie complète, features glissantes et cible calculées."""
    with duckdb.connect(str(chemin), read_only=True) as conn:
        df = conn.execute(f"SELECT * FROM {TABLE}").fetchdf()
    logger.info(f"{len(df):,} lignes chargées depuis {chemin.name}.")
    signaler_anomalies(df)
    df = calculer_features(df)
    df[CIBLE] = calculer_cible(df)
    return df


def signaler_anomalies(df: pd.DataFrame) -> dict[str, int]:
    """Compte les valeurs hors bornes physiques et les types de machine inconnus."""
    anomalies = {
        col: int((df[col].notna() & ~df[col].between(bas, haut)).sum()) for col, (bas, haut) in BORNES.items() if col in df
    }
    if "type_machine" in df:
        anomalies["type_machine"] = int((~df["type_machine"].isin(TYPES_MACHINE)).sum())
    anomalies = {col: n for col, n in anomalies.items() if n}
    if anomalies:
        logger.warning(f"Valeurs hors bornes : {anomalies}")
    return anomalies


def calculer_cible(df: pd.DataFrame, horizon: pd.Timedelta = HORIZON, modes: tuple = MODES_PANNE_PREDICTIBLES) -> np.ndarray:
    """1 si une panne prévisible survient sur la machine dans l'horizon [t, t + horizon[, sinon 0."""
    cible = np.zeros(len(df), dtype=int)
    est_panne = ((df["panne"] == 1) & df["mode_panne"].isin(modes)).to_numpy()
    horodatages = df["timestamp"].to_numpy()
    for positions in df.groupby("id_machine", sort=False).indices.values():
        t = horodatages[positions]
        pannes = np.sort(t[est_panne[positions]])
        if len(pannes) == 0:
            continue
        suivante = np.searchsorted(pannes, t)
        dans_horizon = suivante < len(pannes)
        delai = pannes[np.minimum(suivante, len(pannes) - 1)] - t
        cible[positions] = dans_horizon & (delai < horizon.to_timedelta64())
    return cible


def decouper_pli(df: pd.DataFrame, debut: pd.Timestamp, fin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entraînement avant `debut`, validation sur [debut, fin[, machines réservées exclues.
    Tampon d'un horizon avant chaque coupure : la cible de ces lignes dépend de pannes de la période suivante.
    """
    t = df["timestamp"]
    reservee = df["id_machine"].isin(MACHINES_RESERVEES)
    train = df[(t < debut - HORIZON) & ~reservee]
    validation = df[(t >= debut) & (t < fin - HORIZON) & ~reservee]
    return train, validation


def decouper(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train avant octobre 2024, validation jusqu'à fin 2024, test sur 2025 (toutes machines)."""
    train, validation = decouper_pli(df, DATE_VALIDATION, DATE_TEST)
    return train, validation, df[df["timestamp"] >= DATE_TEST]
