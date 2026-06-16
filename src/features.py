"""
Feature engineering temporel — source unique de vérité pour l'entraînement ET l'inférence.

3 features rolling haute valeur, fenêtre 24h (48 pas × 30 min) :
  - tool_wear_delta_24h   : taux d'accumulation de l'usure → détecte la dégradation rapide
  - vibration_max_24h     : pire vibration récente → plus robuste qu'un point instantané
  - process_temp_max_24h  : pic thermique récent → stress mécanique cumulé

Règle : toujours clip(lower=0) sur les deltas pour absorber les remises à zéro post-maintenance.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_24H = 48          # 48 pas × 30 min = 24 h
TEMPORAL_FEATURES = [
    "tool_wear_delta_24h",
    "vibration_max_24h",
    "process_temp_max_24h",
]


def enrich_training_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features temporelles sur le dataset d'entraînement complet.
    Requiert les colonnes : machine_id, timestamp, tool_wear, vibration, process_temperature.
    Chaque machine est traitée indépendamment via groupby.
    """
    logger.info("Calcul des features temporelles sur le dataset d'entraînement...")
    df = df.sort_values(["machine_id", "timestamp"]).copy()

    grp = df.groupby("machine_id", sort=False)

    # Δ usure sur 24h : wear_t - wear_(t-48), clampé à 0 (absorbe les resets post-maintenance)
    df["tool_wear_delta_24h"] = (
        grp["tool_wear"]
        .transform(lambda x: (x - x.shift(WINDOW_24H)).clip(lower=0))
        .fillna(0.0)
    )

    # Max vibration sur la fenêtre glissante 24h
    df["vibration_max_24h"] = grp["vibration"].transform(
        lambda x: x.rolling(WINDOW_24H, min_periods=1).max()
    )

    # Max température process sur 24h
    df["process_temp_max_24h"] = grp["process_temperature"].transform(
        lambda x: x.rolling(WINDOW_24H, min_periods=1).max()
    )

    logger.info(
        f"Features temporelles calculées : {TEMPORAL_FEATURES} "
        f"sur {df['machine_id'].nunique()} machines."
    )
    return df


def enrich_inference_series(df_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features temporelles sur une série d'une seule machine (chemin Streamlit).
    df_ts doit être trié par timestamp ASC et ne contenir qu'une seule machine.
    """
    df = df_ts.copy()
    df["tool_wear_delta_24h"] = (
        (df["tool_wear"] - df["tool_wear"].shift(WINDOW_24H)).clip(lower=0).fillna(0.0)
    )
    df["vibration_max_24h"] = (
        df["vibration"].rolling(WINDOW_24H, min_periods=1).max()
    )
    df["process_temp_max_24h"] = (
        df["process_temperature"].rolling(WINDOW_24H, min_periods=1).max()
    )
    return df


def enrich_inference_point(current: dict, history_df: pd.DataFrame) -> dict:
    """
    Calcule les features temporelles pour un point unique (chemin API).
    current       : dict avec les features courantes (les 15 features brutes)
    history_df    : DataFrame des dernières 48 lignes de la machine (ASC), peut être vide
    Retourne un dict enrichi avec les 3 features temporelles.
    """
    result = dict(current)

    if history_df.empty:
        result["tool_wear_delta_24h"] = 0.0
        result["vibration_max_24h"] = float(current.get("vibration", 0.0))
        result["process_temp_max_24h"] = float(current.get("process_temperature", 0.0))
        return result

    # Δ usure : wear courant - wear il y a 24h (première ligne de l'historique)
    oldest_wear = float(history_df["tool_wear"].iloc[0])
    current_wear = float(current.get("tool_wear", 0.0))
    result["tool_wear_delta_24h"] = max(0.0, current_wear - oldest_wear)

    # Max vibration : max sur historique + valeur courante
    hist_vibr = history_df["vibration"].dropna().tolist()
    result["vibration_max_24h"] = float(
        np.max(hist_vibr + [current.get("vibration", 0.0)])
    )

    # Max température : max sur historique + valeur courante
    hist_temp = history_df["process_temperature"].dropna().tolist()
    result["process_temp_max_24h"] = float(
        np.max(hist_temp + [current.get("process_temperature", 0.0)])
    )

    return result
