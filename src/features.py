import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FENETRE_4H = 8  # 8 pas × 30 min = 4 h — accélération court terme
FENETRE_24H = 48  # 48 pas × 30 min = 24 h
VARIABLES_TEMPORELLES = [
    "usure_outil_delta_24h",
    "usure_outil_delta_4h",
    "vibration_max_24h",
    "vibration_moyenne_24h",
    "vibration_ecart_type_24h",
    "vibration_delta_4h",
    "temperature_process_max_24h",
    "temperature_process_moyenne_24h",
    "temperature_process_delta_4h",
    "ecart_thermique",
    "ecart_thermique_max_24h",
    "poussiere_moyenne_24h",
    "poussiere_max_24h",
    "couple_max_24h",
    "encrassement_max_24h",
]


def _delta(series: pd.Series, window: int) -> pd.Series:
    """usure/vibration/temp_t - valeur_(t-window), clampé à 0 (absorbe les resets post-maintenance)."""
    return (series - series.shift(window)).clip(lower=0).fillna(0.0)


def enrichir_df_entrainement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features temporelles sur le jeu_donnees d'entraînement complet.
    Requiert les colonnes : id_machine, timestamp, usure_outil, vibration, temperature_process,
    temperature_ambiante, concentration_poussiere, couple, encrassement.
    Chaque machine est traitée indépendamment via groupby.
    """
    logger.info("Calcul des features temporelles sur le jeu_donnees d'entraînement...")
    df = df.sort_values(["id_machine", "timestamp"]).copy()

    grp = df.groupby("id_machine", sort=False)

    df["usure_outil_delta_24h"] = grp["usure_outil"].transform(lambda x: _delta(x, FENETRE_24H))
    df["usure_outil_delta_4h"] = grp["usure_outil"].transform(lambda x: _delta(x, FENETRE_4H))

    df["vibration_max_24h"] = grp["vibration"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).max())
    df["vibration_moyenne_24h"] = grp["vibration"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).mean())
    df["vibration_ecart_type_24h"] = grp["vibration"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).std().fillna(0.0))
    df["vibration_delta_4h"] = grp["vibration"].transform(lambda x: _delta(x, FENETRE_4H))

    df["temperature_process_max_24h"] = grp["temperature_process"].transform(
        lambda x: x.rolling(FENETRE_24H, min_periods=1).max()
    )
    df["temperature_process_moyenne_24h"] = grp["temperature_process"].transform(
        lambda x: x.rolling(FENETRE_24H, min_periods=1).mean()
    )
    df["temperature_process_delta_4h"] = grp["temperature_process"].transform(lambda x: _delta(x, FENETRE_4H))

    # Écart thermique process/ambiant : le générateur définit HDF sur cette grandeur
    # (moyennée 4h), mais elle n'était pas exposée comme feature au modèle — gain
    # quasi gratuit sur le mode le moins bien détecté (point 19).
    df["ecart_thermique"] = df["temperature_process"] - df["temperature_ambiante"]
    # Pic 24h de l'écart thermique — HDF vérifié sous-détecté (68 % de recall) avec
    # seulement la valeur instantanée et les agrégats sur temperature_process seul ;
    # le pic sur ecart_thermique lui-même colle davantage au mécanisme réel du générateur
    # (surchauffe soutenue). Gain mesuré : +7 pts de recall HDF. Cf. docs/model-decisions.md.
    df["ecart_thermique_max_24h"] = grp["ecart_thermique"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).max())

    # Poussière cumulée sur 24h : signal précurseur de l'encrassement (chaîne
    # poussière → HDF, cf. docs/data-generation.md), pas juste la lecture instantanée.
    df["poussiere_moyenne_24h"] = grp["concentration_poussiere"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).mean())
    df["poussiere_max_24h"] = grp["concentration_poussiere"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).max())

    # Pic de couple sur 24h : le couple instantané est bruyant (varie avec l'activité
    # à chaque pas) et noyait le signal précurseur OSF (couple élevé combiné à
    # l'usure) — vérifié sous-détecté (47 % de recall) avec le couple brut seul.
    # Gain mesuré : +10 pts de recall OSF. Cf. docs/model-decisions.md.
    df["couple_max_24h"] = grp["couple"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).max())

    # Encrassement cumulé sur 24h : présent dans les données depuis toujours
    # (alimente la chaleur du process, cf. docs/data-generation.md) mais jamais
    # exposé au modèle avant ce commit. Gain mesuré : +14 pts HDF, +10 pts OSF,
    # +6 pts global. Cf. docs/model-decisions.md.
    df["encrassement_max_24h"] = grp["encrassement"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).max())

    # Cible retenue pour l'entraînement (cf. config.yaml).
    df["cible_h120_predictible"] = calculer_cible_horizon(df, heures_horizon=120, modes_panne=MODES_PANNE_PREDICTIBLES)

    logger.info(f"Features temporelles calculées : {VARIABLES_TEMPORELLES} sur {df['id_machine'].nunique()} machines.")
    return df


# Les autres causes de panne sont non prévisibles, par nature (point 18 — vérifié
# empiriquement : ratios signal pré-panne / baseline pour PWF et RNF entre 0,98 et
# 1,16 selon le capteur, contre 2 à 4x pour TWF/HDF/OSF. Cf. docs/model-decisions.md).
MODES_PANNE_PREDICTIBLES = ("TWF", "HDF", "OSF")


def calculer_cible_horizon(df: pd.DataFrame, heures_horizon: int = 24, modes_panne: tuple | None = None) -> pd.Series:
    """
    Label ML à horizon : 1 sur les lignes situées dans les `heures_horizon` précédant
    une panne brute (panne==1), sinon 0. Découplé de la génération physique :
    la panne réelle reste un événement rare et calibré, ce label élargit juste la
    fenêtre exploitable par le modèle pour apprendre le signal de dégradation qui
    précède l'événement.

    `modes_panne` : si fourni, restreint la cible aux pannes dont la cause appartient
    à cette liste (ex. MODES_PANNE_PREDICTIBLES) — les autres causes ne contribuent pas
    au label, plutôt que d'injecter des fenêtres positives sans signal apprenable.
    """
    pas_horizon = heures_horizon * 2  # pas de 30 min
    index_original = df.index
    df_trie = df.sort_values(["id_machine", "timestamp"])
    panne = df_trie["panne"]
    if modes_panne is not None:
        panne = panne.where(df_trie["mode_panne"].isin(modes_panne), 0)
    cible_horizon = panne.groupby(df_trie["id_machine"], sort=False).transform(
        lambda x: x[::-1].rolling(pas_horizon, min_periods=1).max()[::-1]
    )
    return cible_horizon.reindex(index_original).astype(int)


def enrichir_serie_inference(df_serie: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features temporelles sur une série d'une seule machine (chemin Streamlit).
    df_serie doit être trié par timestamp ASC et ne contenir qu'une seule machine.
    """
    df = df_serie.copy()
    df["usure_outil_delta_24h"] = _delta(df["usure_outil"], FENETRE_24H)
    df["usure_outil_delta_4h"] = _delta(df["usure_outil"], FENETRE_4H)

    df["vibration_max_24h"] = df["vibration"].rolling(FENETRE_24H, min_periods=1).max()
    df["vibration_moyenne_24h"] = df["vibration"].rolling(FENETRE_24H, min_periods=1).mean()
    df["vibration_ecart_type_24h"] = df["vibration"].rolling(FENETRE_24H, min_periods=1).std().fillna(0.0)
    df["vibration_delta_4h"] = _delta(df["vibration"], FENETRE_4H)

    df["temperature_process_max_24h"] = df["temperature_process"].rolling(FENETRE_24H, min_periods=1).max()
    df["temperature_process_moyenne_24h"] = df["temperature_process"].rolling(FENETRE_24H, min_periods=1).mean()
    df["temperature_process_delta_4h"] = _delta(df["temperature_process"], FENETRE_4H)

    df["ecart_thermique"] = df["temperature_process"] - df["temperature_ambiante"]
    df["ecart_thermique_max_24h"] = df["ecart_thermique"].rolling(FENETRE_24H, min_periods=1).max()

    df["poussiere_moyenne_24h"] = df["concentration_poussiere"].rolling(FENETRE_24H, min_periods=1).mean()
    df["poussiere_max_24h"] = df["concentration_poussiere"].rolling(FENETRE_24H, min_periods=1).max()

    df["couple_max_24h"] = df["couple"].rolling(FENETRE_24H, min_periods=1).max()
    df["encrassement_max_24h"] = df["encrassement"].rolling(FENETRE_24H, min_periods=1).max()
    return df


def enrichir_point_inference(current: dict, history_df: pd.DataFrame) -> dict:
    """
    Calcule les features temporelles pour un point unique (chemin API).
    current       : dict avec les features courantes (les 16 features brutes)
    history_df    : DataFrame des 48 dernières lignes de la machine (24h, ASC :
                     usure_outil, vibration, temperature_process, concentration_poussiere,
                     couple, temperature_ambiante, encrassement), peut être vide.
    Retourne un dict enrichi avec les features temporelles.
    """
    result = dict(current)
    usure_actuelle = float(current.get("usure_outil", 0.0))
    vibration_actuelle = float(current.get("vibration", 0.0))
    temperature_actuelle = float(current.get("temperature_process", 0.0))
    ambiante_actuelle = float(current.get("temperature_ambiante", 0.0))
    poussiere_actuelle = float(current.get("concentration_poussiere", 0.0))
    couple_actuel = float(current.get("couple", 0.0))
    encrassement_actuel = float(current.get("encrassement", 0.0))
    ecart_thermique_actuel = temperature_actuelle - ambiante_actuelle

    # Ne dépend pas de l'historique — calculé dans tous les cas.
    result["ecart_thermique"] = ecart_thermique_actuel

    if history_df.empty:
        result["usure_outil_delta_24h"] = 0.0
        result["usure_outil_delta_4h"] = 0.0
        result["vibration_max_24h"] = vibration_actuelle
        result["vibration_moyenne_24h"] = vibration_actuelle
        result["vibration_ecart_type_24h"] = 0.0
        result["vibration_delta_4h"] = 0.0
        result["temperature_process_max_24h"] = temperature_actuelle
        result["temperature_process_moyenne_24h"] = temperature_actuelle
        result["temperature_process_delta_4h"] = 0.0
        result["ecart_thermique_max_24h"] = ecart_thermique_actuel
        result["poussiere_moyenne_24h"] = poussiere_actuelle
        result["poussiere_max_24h"] = poussiere_actuelle
        result["couple_max_24h"] = couple_actuel
        result["encrassement_max_24h"] = encrassement_actuel
        return result

    def _delta_historique(col: str, current_value: float, window: int) -> float:
        hist = history_df[col].dropna()
        if len(hist) < window:
            return 0.0
        return max(0.0, current_value - float(hist.iloc[-window]))

    def _stats_24h(historique: list[float], valeur_actuelle: float) -> dict[str, float]:
        """Moyenne/écart-type/max sur les 24h réellement disponibles — dégrade
        proprement avec moins d'historique plutôt que de planter."""
        serie = (historique + [valeur_actuelle])[-FENETRE_24H:]
        return {
            "moyenne": float(np.mean(serie)),
            "ecart_type": float(np.std(serie)) if len(serie) > 1 else 0.0,
            "max": float(np.max(serie)),
        }

    result["usure_outil_delta_24h"] = _delta_historique("usure_outil", usure_actuelle, FENETRE_24H)
    result["usure_outil_delta_4h"] = _delta_historique("usure_outil", usure_actuelle, FENETRE_4H)

    historique_vibration = history_df["vibration"].dropna().tolist()
    stats_vibration = _stats_24h(historique_vibration, vibration_actuelle)
    result["vibration_max_24h"] = stats_vibration["max"]
    result["vibration_moyenne_24h"] = stats_vibration["moyenne"]
    result["vibration_ecart_type_24h"] = stats_vibration["ecart_type"]
    result["vibration_delta_4h"] = _delta_historique("vibration", vibration_actuelle, FENETRE_4H)

    historique_temperature = history_df["temperature_process"].dropna().tolist()
    stats_temperature = _stats_24h(historique_temperature, temperature_actuelle)
    result["temperature_process_max_24h"] = stats_temperature["max"]
    result["temperature_process_moyenne_24h"] = stats_temperature["moyenne"]
    result["temperature_process_delta_4h"] = _delta_historique("temperature_process", temperature_actuelle, FENETRE_4H)

    historique_poussiere = (
        history_df["concentration_poussiere"].dropna().tolist() if "concentration_poussiere" in history_df else []
    )
    stats_poussiere = _stats_24h(historique_poussiere, poussiere_actuelle)
    result["poussiere_moyenne_24h"] = stats_poussiere["moyenne"]
    result["poussiere_max_24h"] = stats_poussiere["max"]

    # Couple : pic sur historique + valeur courante (signal précurseur OSF, point
    # "amélioration détection HDF/OSF" — cf. docs/model-decisions.md).
    historique_couple = history_df["couple"].dropna().tolist() + [couple_actuel] if "couple" in history_df else [couple_actuel]
    result["couple_max_24h"] = float(np.max(historique_couple[-FENETRE_24H:]))

    # Encrassement : pic sur 24h (brut + historique), même patron que couple_max_24h.
    historique_encrassement = history_df["encrassement"].dropna().tolist() if "encrassement" in history_df else []
    result["encrassement_max_24h"] = float(np.max(historique_encrassement[-FENETRE_24H:] + [encrassement_actuel]))

    if "temperature_process" in history_df and "temperature_ambiante" in history_df:
        historique_ambiante = history_df["temperature_ambiante"].dropna()
        historique_process = history_df["temperature_process"].dropna()
        historique_ecart_thermique = (historique_process - historique_ambiante).dropna().tolist()
    else:
        historique_ecart_thermique = []
    stats_ecart_thermique = _stats_24h(historique_ecart_thermique, ecart_thermique_actuel)
    result["ecart_thermique_max_24h"] = stats_ecart_thermique["max"]

    return result
