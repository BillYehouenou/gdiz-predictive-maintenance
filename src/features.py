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
    "delta_temp",
    "dust_mean_24h",
    "dust_max_24h",
    "torque_max_24h",
    "delta_temp_max_24h",
]


def _delta(series: pd.Series, window: int) -> pd.Series:
    """wear/vibration/temp_t - valeur_(t-window), clampé à 0 (absorbe les resets post-maintenance)."""
    return (series - series.shift(window)).clip(lower=0).fillna(0.0)


def enrich_training_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features temporelles sur le dataset d'entraînement complet.
    Requiert les colonnes : machine_id, timestamp, tool_wear, vibration, process_temperature,
    ambient_temperature, dust_concentration, torque.
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

    # Écart thermique process/ambiant : le générateur définit HDF sur cette grandeur
    # (moyennée 4h), mais elle n'était pas exposée comme feature au modèle — gain
    # quasi gratuit sur le mode le moins bien détecté (point 19).
    df["delta_temp"] = df["process_temperature"] - df["ambient_temperature"]
    # Pic 24h de l'écart thermique — HDF vérifié sous-détecté (68 % de recall) avec
    # seulement la valeur instantanée et les agrégats sur process_temperature seul ;
    # le pic sur delta_temp lui-même colle davantage au mécanisme réel du générateur
    # (surchauffe soutenue). Gain mesuré : +7 pts de recall HDF. Cf. docs/model-decisions.md.
    df["delta_temp_max_24h"] = grp["delta_temp"].transform(lambda x: x.rolling(WINDOW_24H, min_periods=1).max())

    # Poussière cumulée sur 24h : signal précurseur de l'encrassement (chaîne
    # poussière → HDF, cf. docs/data-generation.md), pas juste la lecture instantanée.
    df["dust_mean_24h"] = grp["dust_concentration"].transform(lambda x: x.rolling(WINDOW_24H, min_periods=1).mean())
    df["dust_max_24h"] = grp["dust_concentration"].transform(lambda x: x.rolling(WINDOW_24H, min_periods=1).max())

    # Pic de couple sur 24h : le couple instantané est bruyant (varie avec l'activité
    # à chaque pas) et noyait le signal précurseur OSF (couple élevé combiné à
    # l'usure) — vérifié sous-détecté (47 % de recall) avec le couple brut seul.
    # Gain mesuré : +10 pts de recall OSF. Cf. docs/model-decisions.md.
    df["torque_max_24h"] = grp["torque"].transform(lambda x: x.rolling(WINDOW_24H, min_periods=1).max())

    # Label ML à horizon : "panne dans les prochaines 24h/48h" — élargit la fenêtre
    # positive exploitable par le modèle au-delà de l'instant exact de la panne brute.
    df["target_h24"] = compute_horizon_target(df, horizon_hours=24)
    df["target_h48"] = compute_horizon_target(df, horizon_hours=48)
    # Cible retenue pour l'entraînement (cf. config.yaml).
    df["target_h120_predictable"] = compute_horizon_target(df, horizon_hours=120, failure_modes=PREDICTABLE_FAILURE_MODES)
    # Horizons différenciés par mode (point 15) — calculée pour rester disponible et
    # testable, mais PAS la cible active : testée et invalidée empiriquement sur ce
    # dataset (recall événementiel en validation inférieur à l'horizon unique dans
    # toutes les configurations essayées). Cf. docs/model-decisions.md.
    df["target_multi_horizon"] = compute_multi_horizon_target(df, HORIZON_BY_FAILURE_MODE)

    logger.info(f"Features temporelles calculées : {TEMPORAL_FEATURES} sur {df['machine_id'].nunique()} machines.")
    return df


# Les autres causes de panne sont non prévisibles, par nature (point 18 — vérifié
# empiriquement : ratios signal pré-panne / baseline pour PWF et RNF entre 0,98 et
# 1,16 selon le capteur, contre 2 à 4x pour TWF/HDF/OSF. Cf. docs/model-decisions.md).
PREDICTABLE_FAILURE_MODES = ("TWF", "HDF", "OSF")

# Horizons différenciés par mode de panne (point 15) — hypothèse de la roadmap :
# HDF/OSF perdraient en précision au-delà de 72h. Testée avec plusieurs
# combinaisons (cf. docs/model-decisions.md) via target_multi_horizon ci-dessus :
# AUCUNE ne bat l'horizon unique de 120h sur le recall événementiel en validation
# sur ce dataset. Valeur ci-dessous = meilleure combinaison testée, gardée
# disponible pour ré-essayer avec un futur générateur/modèle, pas activée par défaut.
HORIZON_BY_FAILURE_MODE = {"TWF": 120, "HDF": 72, "OSF": 72}


def compute_multi_horizon_target(df: pd.DataFrame, horizon_by_mode: dict[str, int]) -> pd.Series:
    """
    Cible combinée : 1 si la ligne est dans l'horizon propre à AU MOINS UN mode de
    panne prédictible à venir, chaque mode évalué avec sa propre fenêtre plutôt
    qu'un horizon unique appliqué à tous.
    """
    combined = pd.Series(0, index=df.index)
    for mode, horizon_hours in horizon_by_mode.items():
        mode_target = compute_horizon_target(df, horizon_hours=horizon_hours, failure_modes=(mode,))
        combined = combined | mode_target
    return combined.astype(int)


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


def compute_failure_cause_label(
    df: pd.DataFrame, horizon_hours: int = 120, failure_modes: tuple = PREDICTABLE_FAILURE_MODES
) -> pd.Series:
    """
    Cible du diagnostic de cause (étage 2, point 14) : pour chaque ligne, la cause de
    la PROCHAINE panne prédictible dans les `horizon_hours` à venir (None si aucune).
    Entraîné uniquement sur les lignes où le label de l'étage 1 (target_h120_predictable)
    vaut 1 — pas un détecteur, un diagnostic qui s'exécute après qu'une alerte a déjà
    été levée. Cf. docs/model-decisions.md pour pourquoi ce n'est pas censé améliorer
    le recall de l'étage 1 (testé et écarté : classifieurs séparés par cause, horizons
    différenciés, pondération des exemples OSF — aucun ne bat le modèle mutualisé).
    """
    horizon_steps = horizon_hours * 2
    df_sorted = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=False)
    result = np.full(len(df_sorted), None, dtype=object)

    for _, g in df_sorted.groupby("machine_id", sort=False):
        start = g.index[0]
        n = len(g)
        modes_arr = g["failure_mode"].to_numpy()
        is_fail = (g["target"].to_numpy() == 1) & np.isin(modes_arr, failure_modes)
        fail_positions = np.flatnonzero(is_fail)
        if len(fail_positions) == 0:
            continue

        local_positions = np.arange(n)
        insertion = np.searchsorted(fail_positions, local_positions, side="left")
        has_next = insertion < len(fail_positions)
        next_fail_position = np.where(has_next, fail_positions[np.clip(insertion, 0, len(fail_positions) - 1)], -1)
        distance = next_fail_position - local_positions
        within_horizon = has_next & (distance >= 0) & (distance <= horizon_steps)
        result[start : start + n] = np.where(within_horizon, modes_arr[np.clip(next_fail_position, 0, n - 1)], None)

    return pd.Series(result, index=df_sorted["index"]).reindex(df.index)


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

    df["delta_temp"] = df["process_temperature"] - df["ambient_temperature"]
    df["delta_temp_max_24h"] = df["delta_temp"].rolling(WINDOW_24H, min_periods=1).max()
    df["dust_mean_24h"] = df["dust_concentration"].rolling(WINDOW_24H, min_periods=1).mean()
    df["dust_max_24h"] = df["dust_concentration"].rolling(WINDOW_24H, min_periods=1).max()
    df["torque_max_24h"] = df["torque"].rolling(WINDOW_24H, min_periods=1).max()
    return df


def enrich_inference_point(current: dict, history_df: pd.DataFrame) -> dict:
    """
    Calcule les features temporelles pour un point unique (chemin API).
    current       : dict avec les features courantes (les 15 features brutes)
    history_df    : DataFrame des dernières 48 lignes de la machine (ASC : tool_wear,
                     vibration, process_temperature, dust_concentration, torque,
                     ambient_temperature), peut être vide.
    Retourne un dict enrichi avec les features temporelles.
    """
    result = dict(current)
    current_wear = float(current.get("tool_wear", 0.0))
    current_vibr = float(current.get("vibration", 0.0))
    current_temp = float(current.get("process_temperature", 0.0))
    current_ambient = float(current.get("ambient_temperature", 0.0))
    current_dust = float(current.get("dust_concentration", 0.0))
    current_torque = float(current.get("torque", 0.0))
    current_delta_temp = current_temp - current_ambient

    # Ne dépend pas de l'historique — calculé dans tous les cas.
    result["delta_temp"] = current_delta_temp

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
        result["delta_temp_max_24h"] = current_delta_temp
        result["dust_mean_24h"] = current_dust
        result["dust_max_24h"] = current_dust
        result["torque_max_24h"] = current_torque
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

    # Poussière : max / moyenne sur historique + valeur courante
    hist_dust = (
        history_df["dust_concentration"].dropna().tolist() + [current_dust]
        if "dust_concentration" in history_df
        else [current_dust]
    )
    result["dust_mean_24h"] = float(np.mean(hist_dust))
    result["dust_max_24h"] = float(np.max(hist_dust))

    # Couple : pic sur historique + valeur courante (signal précurseur OSF, point
    # "amélioration détection HDF/OSF" — cf. docs/model-decisions.md).
    hist_torque = history_df["torque"].dropna().tolist() + [current_torque] if "torque" in history_df else [current_torque]
    result["torque_max_24h"] = float(np.max(hist_torque))

    # Écart thermique : pic sur historique + valeur courante.
    if "process_temperature" in history_df and "ambient_temperature" in history_df:
        hist_ambient = history_df["ambient_temperature"].dropna()
        hist_process = history_df["process_temperature"].dropna()
        hist_delta_temp = (hist_process - hist_ambient).dropna().tolist() + [current_delta_temp]
    else:
        hist_delta_temp = [current_delta_temp]
    result["delta_temp_max_24h"] = float(np.max(hist_delta_temp))

    return result
