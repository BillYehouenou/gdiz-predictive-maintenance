"""
Données du modèle : chargement de la télémétrie, features glissantes (une seule fonction pour l'entraînement, le scoring
et l'API), cible à 120 h et découpage train / validation / test.
"""

import logging
import pathlib

import duckdb
import numpy as np
import pandas as pd

from src.config import CHEMIN_DB, CIBLE, PAS, TABLE

logger = logging.getLogger(__name__)

# Mesures brutes vues par le modèle (capteurs + compteurs GMAO) et features glissantes sur 24 h et 4 h
FEATURES_BRUTES = ["type_machine", "temperature_ambiante", "temperature_process", "vitesse_rotation", "couple",
                   "niveau_activite", "vibration", "humidite", "concentration_poussiere", "precipitation_mm",
                   "indicateur_coupure", "niveau_tension", "heures_depuis_changement_outil",
                   "heures_depuis_maintenance"]  # fmt: skip
CAPTEURS = ["vibration", "temperature_process", "ecart_thermique", "couple", "concentration_poussiere"]
FEATURES_GLISSANTES = [f"{c}_{stat}" for c in CAPTEURS for stat in ("moyenne_24h", "max_24h", "ecart_type_24h", "hausse_4h")]
FEATURES = FEATURES_BRUTES + ["ecart_thermique"] + FEATURES_GLISSANTES
TYPES_MACHINE = ["L", "M", "H"]
HISTORIQUE = pd.Timedelta("24h")  # plus longue fenêtre : historique à fournir pour prédire un point
LIGNES_HISTORIQUE = int(HISTORIQUE / PAS)

MODES_PANNE_PREDICTIBLES = ("TWF", "HDF", "OSF")  # PWF et RNF n'ont pas de signe avant-coureur
HORIZON = pd.Timedelta("120h")
DATE_VALIDATION, DATE_TEST = pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-01")
# Fenêtres de validation pour comparer les modèles : chaque trimestre 2024 après le premier
PLIS = [(pd.Timestamp("2024-04-01"), pd.Timestamp("2024-07-01")), (pd.Timestamp("2024-07-01"), DATE_VALIDATION),
        (DATE_VALIDATION, DATE_TEST)]  # fmt: skip
# Une machine sur cinq n'est vue qu'au test : mesure la généralisation à une machine neuve
MACHINES_RESERVEES = [f"GDIZ_TEX_M_{i:02d}" for i in range(5, 51, 5)]
# Bornes physiques larges : une valeur hors bornes est signalée, jamais bloquante
BORNES = {"temperature_process": (-50, 200), "temperature_ambiante": (-20, 60), "vitesse_rotation": (-1000, 5000),
          "couple": (-100, 500), "vibration": (-10, 50), "humidite": (-20, 150), "concentration_poussiere": (-100, 2000),
          "precipitation_mm": (0, 200), "niveau_tension": (0, 400), "niveau_activite": (-0.5, 1.5),
          "heures_depuis_changement_outil": (0, 50_000), "heures_depuis_maintenance": (0, 50_000)}  # fmt: skip


def calculer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features glissantes. Retourne le DataFrame trié par (id_machine, timestamp), index réinitialisé."""
    df = df.sort_values(["id_machine", "timestamp"]).reset_index(drop=True)
    df["ecart_thermique"] = df["temperature_process"] - df["temperature_ambiante"]
    groupes = df.set_index("timestamp").groupby("id_machine", sort=False)[CAPTEURS]
    fenetre_24h = groupes.rolling(HISTORIQUE, min_periods=1)
    moyenne, maximum, ecart_type = fenetre_24h.mean(), fenetre_24h.max(), fenetre_24h.std()
    minimum_4h = groupes.rolling("4h", min_periods=1).min()
    for c in CAPTEURS:
        df[f"{c}_moyenne_24h"] = moyenne[c].to_numpy()
        df[f"{c}_max_24h"] = maximum[c].to_numpy()
        df[f"{c}_ecart_type_24h"] = ecart_type[c].to_numpy()
        df[f"{c}_hausse_4h"] = df[c] - minimum_4h[c].to_numpy()
    return df


def preparer_X(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes du modèle. LightGBM gère nativement les NaN et la catégorie type_machine."""
    X = df[FEATURES].copy()
    X["type_machine"] = pd.Categorical(X["type_machine"], categories=TYPES_MACHINE)
    return X


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
    anomalies = {c: int((df[c].notna() & ~df[c].between(b, h)).sum()) for c, (b, h) in BORNES.items() if c in df}
    if "type_machine" in df:
        anomalies["type_machine"] = int((~df["type_machine"].isin(TYPES_MACHINE)).sum())
    anomalies = {c: n for c, n in anomalies.items() if n}
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
        if len(pannes):
            suivante = np.searchsorted(pannes, t)
            delai = pannes[np.minimum(suivante, len(pannes) - 1)] - t
            cible[positions] = (suivante < len(pannes)) & (delai < horizon.to_timedelta64())
    return cible


def decouper_pli(df: pd.DataFrame, debut: pd.Timestamp, fin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entraînement avant `debut`, validation sur [debut, fin[, machines réservées exclues.
    Tampon d'un horizon avant chaque coupure : la cible de ces lignes dépend de pannes de la période suivante.
    """
    t, reservee = df["timestamp"], df["id_machine"].isin(MACHINES_RESERVEES)
    return df[(t < debut - HORIZON) & ~reservee], df[(t >= debut) & (t < fin - HORIZON) & ~reservee]


def decouper(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train avant octobre 2024, validation jusqu'à fin 2024, test sur 2025 (toutes machines)."""
    train, validation = decouper_pli(df, DATE_VALIDATION, DATE_TEST)
    return train, validation, df[df["timestamp"] >= DATE_TEST]
