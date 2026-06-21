import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_4H = 8  # 8 pas × 30 min = 4 h — accélération court terme
WINDOW_24H = 48  # 48 pas × 30 min = 24 h
TEMPORAL_FEATURES = [
    "tool_wear_delta_24h",
    "tool_wear_delta_4h",
    "vibration_max_24h",
    "vibration_mean_24h",
    "vibration_std_24h",
    "vibration_delta_4h",
    "process_temp_max_24h",
    "process_temp_mean_24h",
    "process_temp_delta_4h",
]


def _delta(series: pd.Series, window: int) -> pd.Series:
    """wear/vibration/temp_t - valeur_(t-window), clampé à 0 (absorbe les resets post-maintenance)."""
    return (series - series.shift(window)).clip(lower=0).fillna(0.0)


def enrich_training_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features temporelles sur le dataset d'entraînement complet.
    Requiert les colonnes : machine_id, timestamp, tool_wear, vibration, process_temperature.
    Chaque machine est traitée indépendamment via groupby.
    """
    logger.info("Calcul des features temporelles sur le dataset d'entraînement...")
    df = df.sort_values(["machine_id", "timestamp"]).copy()

    grp = df.groupby("machine_id", sort=False)

    df["tool_wear_delta_24h"] = grp["tool_wear"].transform(lambda x: _delta(x, WINDOW_24H))
    df["tool_wear_delta_4h"] = grp["tool_wear"].transform(lambda x: _delta(x, WINDOW_4H))

    df["vibration_max_24h"] = grp["vibration"].transform(lambda x: x.rolling(WINDOW_24H, min_periods=1).max())
    df["vibration_mean_24h"] = grp["vibration"].transform(lambda x: x.rolling(WINDOW_24H, min_periods=1).mean())
    df["vibration_std_24h"] = grp["vibration"].transform(lambda x: x.rolling(WINDOW_24H, min_periods=1).std().fillna(0.0))
    df["vibration_delta_4h"] = grp["vibration"].transform(lambda x: _delta(x, WINDOW_4H))

    df["process_temp_max_24h"] = grp["process_temperature"].transform(lambda x: x.rolling(WINDOW_24H, min_periods=1).max())
    df["process_temp_mean_24h"] = grp["process_temperature"].transform(lambda x: x.rolling(WINDOW_24H, min_periods=1).mean())
    df["process_temp_delta_4h"] = grp["process_temperature"].transform(lambda x: _delta(x, WINDOW_4H))

    # Label ML à horizon : "panne dans les prochaines 24h/48h" — élargit la fenêtre
    # positive exploitable par le modèle au-delà de l'instant exact de la panne brute.
    df["target_h24"] = compute_horizon_target(df, horizon_hours=24)
    df["target_h48"] = compute_horizon_target(df, horizon_hours=48)
    # Cible retenue pour l'entraînement
    df["target_h120_predictable"] = compute_horizon_target(df, horizon_hours=120, failure_modes=PREDICTABLE_FAILURE_MODES)

    logger.info(f"Features temporelles calculées : {TEMPORAL_FEATURES} sur {df['machine_id'].nunique()} machines.")
    return df


# Les autres causes de panne sont non prévisibles, par nature.
PREDICTABLE_FAILURE_MODES = ("TWF", "HDF", "OSF")


def compute_horizon_target(df: pd.DataFrame, horizon_hours: int = 24, failure_modes: tuple | None = None) -> pd.Series:
    """
    Label ML à horizon : 1 sur les lignes situées dans les `horizon_hours` précédant
    une panne brute (target==1), sinon 0. Découplé de la génération physique :
    la panne réelle reste un événement rare et calibré, ce label élargit juste la
    fenêtre exploitable par le modèle pour apprendre le signal de dégradation qui
    précède l'événement.

    `failure_modes` : si fourni, restreint la cible aux pannes dont la cause appartient
    à cette liste (ex. PREDICTABLE_FAILURE_MODES) — les autres causes ne contribuent pas
    au label, plutôt que d'injecter des fenêtres positives sans signal apprenable.
    """
    horizon_steps = horizon_hours * 2  # pas de 30 min
    original_index = df.index
    df_sorted = df.sort_values(["machine_id", "timestamp"])
    target = df_sorted["target"]
    if failure_modes is not None:
        target = target.where(df_sorted["failure_mode"].isin(failure_modes), 0)
    horizon_target = target.groupby(df_sorted["machine_id"], sort=False).transform(
        lambda x: x[::-1].rolling(horizon_steps, min_periods=1).max()[::-1]
    )
    return horizon_target.reindex(original_index).astype(int)


def enrich_inference_series(df_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features temporelles sur une série d'une seule machine (chemin Streamlit).
    df_ts doit être trié par timestamp ASC et ne contenir qu'une seule machine.
    """
    df = df_ts.copy()
    df["tool_wear_delta_24h"] = _delta(df["tool_wear"], WINDOW_24H)
    df["tool_wear_delta_4h"] = _delta(df["tool_wear"], WINDOW_4H)

    df["vibration_max_24h"] = df["vibration"].rolling(WINDOW_24H, min_periods=1).max()
    df["vibration_mean_24h"] = df["vibration"].rolling(WINDOW_24H, min_periods=1).mean()
    df["vibration_std_24h"] = df["vibration"].rolling(WINDOW_24H, min_periods=1).std().fillna(0.0)
    df["vibration_delta_4h"] = _delta(df["vibration"], WINDOW_4H)

    df["process_temp_max_24h"] = df["process_temperature"].rolling(WINDOW_24H, min_periods=1).max()
    df["process_temp_mean_24h"] = df["process_temperature"].rolling(WINDOW_24H, min_periods=1).mean()
    df["process_temp_delta_4h"] = _delta(df["process_temperature"], WINDOW_4H)
    return df


def enrich_inference_point(current: dict, history_df: pd.DataFrame) -> dict:
    """
    Calcule les features temporelles pour un point unique (chemin API).
    current       : dict avec les features courantes (les 15 features brutes)
    history_df    : DataFrame des dernières 48 lignes de la machine (ASC : tool_wear,
                     vibration, process_temperature), peut être vide.
    Retourne un dict enrichi avec les 9 features temporelles.
    """
    result = dict(current)
    current_wear = float(current.get("tool_wear", 0.0))
    current_vibr = float(current.get("vibration", 0.0))
    current_temp = float(current.get("process_temperature", 0.0))

    if history_df.empty:
        result["tool_wear_delta_24h"] = 0.0
        result["tool_wear_delta_4h"] = 0.0
        result["vibration_max_24h"] = current_vibr
        result["vibration_mean_24h"] = current_vibr
        result["vibration_std_24h"] = 0.0
        result["vibration_delta_4h"] = 0.0
        result["process_temp_max_24h"] = current_temp
        result["process_temp_mean_24h"] = current_temp
        result["process_temp_delta_4h"] = 0.0
        return result

    def _hist_delta(col: str, current_value: float, window: int) -> float:
        hist = history_df[col].dropna()
        if len(hist) < window:
            return 0.0
        return max(0.0, current_value - float(hist.iloc[-window]))

    # Usure : delta 24h (historique complet) et 4h (8 dernières lignes)
    oldest_wear = float(history_df["tool_wear"].iloc[0])
    result["tool_wear_delta_24h"] = max(0.0, current_wear - oldest_wear)
    result["tool_wear_delta_4h"] = _hist_delta("tool_wear", current_wear, WINDOW_4H)

    # Vibration : max / moyenne / écart-type sur historique + valeur courante, delta 4h
    hist_vibr = history_df["vibration"].dropna().tolist() + [current_vibr]
    result["vibration_max_24h"] = float(np.max(hist_vibr))
    result["vibration_mean_24h"] = float(np.mean(hist_vibr))
    result["vibration_std_24h"] = float(np.std(hist_vibr)) if len(hist_vibr) > 1 else 0.0
    result["vibration_delta_4h"] = _hist_delta("vibration", current_vibr, WINDOW_4H)

    # Température process : max / moyenne sur historique + valeur courante, delta 4h
    hist_temp = history_df["process_temperature"].dropna().tolist() + [current_temp]
    result["process_temp_max_24h"] = float(np.max(hist_temp))
    result["process_temp_mean_24h"] = float(np.mean(hist_temp))
    result["process_temp_delta_4h"] = _hist_delta("process_temperature", current_temp, WINDOW_4H)

    return result
