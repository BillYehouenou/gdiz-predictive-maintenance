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
    "poussiere_moyenne_24h",
    "poussiere_max_24h",
    "couple_max_24h",
    "ecart_thermique_max_24h",
]


def _delta(series: pd.Series, window: int) -> pd.Series:
    """wear/vibration/temp_t - valeur_(t-window), clampé à 0 (absorbe les resets post-maintenance)."""
    return (series - series.shift(window)).clip(lower=0).fillna(0.0)


def enrichir_df_entrainement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features temporelles sur le jeu_donnees d'entraînement complet.
    Requiert les colonnes : id_machine, timestamp, usure_outil, vibration, temperature_process,
    temperature_ambiante, concentration_poussiere, couple.
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

    # Label ML à horizon : "panne dans les prochaines 24h/48h" — élargit la fenêtre
    # positive exploitable par le modèle au-delà de l'instant exact de la panne brute.
    df["cible_h24"] = calculer_cible_horizon(df, heures_horizon=24)
    df["cible_h48"] = calculer_cible_horizon(df, heures_horizon=48)
    # Cible retenue pour l'entraînement (cf. config.yaml).
    df["cible_h120_predictible"] = calculer_cible_horizon(df, heures_horizon=120, modes_panne=MODES_PANNE_PREDICTIBLES)
    # Horizons différenciés par mode (point 15) — calculée pour rester disponible et
    # testable, mais PAS la cible active : testée et invalidée empiriquement sur ce
    # jeu_donnees (recall événementiel en validation inférieur à l'horizon unique dans
    # toutes les configurations essayées). Cf. docs/model-decisions.md.
    df["cible_multi_horizon"] = calculer_cible_multi_horizon(df, HORIZON_PAR_MODE_PANNE)

    logger.info(f"Features temporelles calculées : {VARIABLES_TEMPORELLES} sur {df['id_machine'].nunique()} machines.")
    return df


# Les autres causes de panne sont non prévisibles, par nature (point 18 — vérifié
# empiriquement : ratios signal pré-panne / baseline pour PWF et RNF entre 0,98 et
# 1,16 selon le capteur, contre 2 à 4x pour TWF/HDF/OSF. Cf. docs/model-decisions.md).
MODES_PANNE_PREDICTIBLES = ("TWF", "HDF", "OSF")

# Horizons différenciés par mode de panne (point 15) — hypothèse de la roadmap :
# HDF/OSF perdraient en précision au-delà de 72h. Testée avec plusieurs
# combinaisons (cf. docs/model-decisions.md) via cible_multi_horizon ci-dessus :
# AUCUNE ne bat l'horizon unique de 120h sur le recall événementiel en validation
# sur ce jeu_donnees. Valeur ci-dessous = meilleure combinaison testée, gardée
# disponible pour ré-essayer avec un futur générateur/modèle, pas activée par défaut.
HORIZON_PAR_MODE_PANNE = {"TWF": 120, "HDF": 72, "OSF": 72}


def calculer_cible_multi_horizon(df: pd.DataFrame, horizon_par_mode: dict[str, int]) -> pd.Series:
    """
    Cible combinée : 1 si la ligne est dans l'horizon propre à AU MOINS UN mode de
    panne prédictible à venir, chaque mode évalué avec sa propre fenêtre plutôt
    qu'un horizon unique appliqué à tous.
    """
    combined = pd.Series(0, index=df.index)
    for mode, heures_horizon in horizon_par_mode.items():
        mode_target = calculer_cible_horizon(df, heures_horizon=heures_horizon, modes_panne=(mode,))
        combined = combined | mode_target
    return combined.astype(int)


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


def calculer_label_cause_panne(
    df: pd.DataFrame, heures_horizon: int = 120, modes_panne: tuple = MODES_PANNE_PREDICTIBLES
) -> pd.Series:
    """
    Cible du diagnostic de cause (étage 2, point 14) : pour chaque ligne, la cause de
    la PROCHAINE panne prédictible dans les `heures_horizon` à venir (None si aucune).
    Entraîné uniquement sur les lignes où le label de l'étage 1 (cible_h120_predictible)
    vaut 1 — pas un détecteur, un diagnostic qui s'exécute après qu'une alerte a déjà
    été levée. Cf. docs/model-decisions.md pour pourquoi ce n'est pas censé améliorer
    le recall de l'étage 1 (testé et écarté : classifieurs séparés par cause, horizons
    différenciés, pondération des exemples OSF — aucun ne bat le modèle mutualisé).
    """
    pas_horizon = heures_horizon * 2
    df_trie = df.sort_values(["id_machine", "timestamp"]).reset_index(drop=False)
    result = np.full(len(df_trie), None, dtype=object)

    for _, g in df_trie.groupby("id_machine", sort=False):
        start = g.index[0]
        n = len(g)
        tableau_modes = g["mode_panne"].to_numpy()
        est_panne = (g["panne"].to_numpy() == 1) & np.isin(tableau_modes, modes_panne)
        positions_pannes = np.flatnonzero(est_panne)
        if len(positions_pannes) == 0:
            continue

        positions_locales = np.arange(n)
        insertion = np.searchsorted(positions_pannes, positions_locales, side="left")
        a_suivant = insertion < len(positions_pannes)
        position_prochaine_panne = np.where(a_suivant, positions_pannes[np.clip(insertion, 0, len(positions_pannes) - 1)], -1)
        distance = position_prochaine_panne - positions_locales
        dans_horizon = a_suivant & (distance >= 0) & (distance <= pas_horizon)
        result[start : start + n] = np.where(dans_horizon, tableau_modes[np.clip(position_prochaine_panne, 0, n - 1)], None)

    return pd.Series(result, index=df_trie["index"]).reindex(df.index)


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
    return df


def enrichir_point_inference(current: dict, history_df: pd.DataFrame) -> dict:
    """
    Calcule les features temporelles pour un point unique (chemin API).
    current       : dict avec les features courantes (les 15 features brutes)
    history_df    : DataFrame des dernières 48 lignes de la machine (ASC : usure_outil,
                     vibration, temperature_process, concentration_poussiere, couple,
                     temperature_ambiante), peut être vide.
    Retourne un dict enrichi avec les features temporelles.
    """
    result = dict(current)
    usure_actuelle = float(current.get("usure_outil", 0.0))
    vibration_actuelle = float(current.get("vibration", 0.0))
    temperature_actuelle = float(current.get("temperature_process", 0.0))
    ambiante_actuelle = float(current.get("temperature_ambiante", 0.0))
    poussiere_actuelle = float(current.get("concentration_poussiere", 0.0))
    couple_actuel = float(current.get("couple", 0.0))
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
        return result

    def _delta_historique(col: str, current_value: float, window: int) -> float:
        hist = history_df[col].dropna()
        if len(hist) < window:
            return 0.0
        return max(0.0, current_value - float(hist.iloc[-window]))

    # Usure : delta 24h (historique complet) et 4h (8 dernières lignes)
    usure_plus_ancienne = float(history_df["usure_outil"].iloc[0])
    result["usure_outil_delta_24h"] = max(0.0, usure_actuelle - usure_plus_ancienne)
    result["usure_outil_delta_4h"] = _delta_historique("usure_outil", usure_actuelle, FENETRE_4H)

    # Vibration : max / moyenne / écart-type sur historique + valeur courante, delta 4h
    historique_vibration = history_df["vibration"].dropna().tolist() + [vibration_actuelle]
    result["vibration_max_24h"] = float(np.max(historique_vibration))
    result["vibration_moyenne_24h"] = float(np.mean(historique_vibration))
    result["vibration_ecart_type_24h"] = float(np.std(historique_vibration)) if len(historique_vibration) > 1 else 0.0
    result["vibration_delta_4h"] = _delta_historique("vibration", vibration_actuelle, FENETRE_4H)

    # Température process : max / moyenne sur historique + valeur courante, delta 4h
    historique_temperature = history_df["temperature_process"].dropna().tolist() + [temperature_actuelle]
    result["temperature_process_max_24h"] = float(np.max(historique_temperature))
    result["temperature_process_moyenne_24h"] = float(np.mean(historique_temperature))
    result["temperature_process_delta_4h"] = _delta_historique("temperature_process", temperature_actuelle, FENETRE_4H)

    # Poussière : max / moyenne sur historique + valeur courante
    historique_poussiere = (
        history_df["concentration_poussiere"].dropna().tolist() + [poussiere_actuelle]
        if "concentration_poussiere" in history_df
        else [poussiere_actuelle]
    )
    result["poussiere_moyenne_24h"] = float(np.mean(historique_poussiere))
    result["poussiere_max_24h"] = float(np.max(historique_poussiere))

    # Couple : pic sur historique + valeur courante (signal précurseur OSF, point
    # "amélioration détection HDF/OSF" — cf. docs/model-decisions.md).
    historique_couple = history_df["couple"].dropna().tolist() + [couple_actuel] if "couple" in history_df else [couple_actuel]
    result["couple_max_24h"] = float(np.max(historique_couple))

    # Écart thermique : pic sur historique + valeur courante.
    if "temperature_process" in history_df and "temperature_ambiante" in history_df:
        historique_ambiante = history_df["temperature_ambiante"].dropna()
        historique_process = history_df["temperature_process"].dropna()
        historique_ecart_thermique = (historique_process - historique_ambiante).dropna().tolist() + [ecart_thermique_actuel]
    else:
        historique_ecart_thermique = [ecart_thermique_actuel]
    result["ecart_thermique_max_24h"] = float(np.max(historique_ecart_thermique))

    return result
