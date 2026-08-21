import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FENETRE_4H = 8  # 8 pas × 30 min = 4 h — accélération court terme
FENETRE_24H = 48  # 48 pas × 30 min = 24 h
FENETRE_72H = 144  # 144 pas × 30 min = 72 h — lisse le bruit court terme, capte une dérive soutenue
FENETRE_120H = 240  # 240 pas × 30 min = 120 h — aligné sur l'horizon de la cible active
VARIABLES_TEMPORELLES = [
    "usure_outil_delta_24h",
    "usure_outil_delta_4h",
    "vibration_max_24h",
    "vibration_moyenne_24h",
    "vibration_ecart_type_24h",
    "vibration_delta_4h",
    "vibration_moyenne_72h",
    "vibration_ecart_type_72h",
    "vibration_moyenne_120h",
    "temperature_process_max_24h",
    "temperature_process_moyenne_24h",
    "temperature_process_delta_4h",
    "temperature_process_moyenne_72h",
    "temperature_process_moyenne_120h",
    "ecart_thermique",
    "ecart_thermique_max_24h",
    "ecart_thermique_moyenne_72h",
    "ecart_thermique_moyenne_120h",
    "poussiere_moyenne_24h",
    "poussiere_max_24h",
    "poussiere_moyenne_72h",
    "poussiere_moyenne_120h",
    "couple_max_24h",
    "pente_ecart_thermique_72h",
    "pente_vibration_72h",
    "ratio_vibration_couple",
]


def _delta(series: pd.Series, window: int) -> pd.Series:
    """usure/vibration/temp_t - valeur_(t-window), clampé à 0 (absorbe les resets post-maintenance)."""
    return (series - series.shift(window)).clip(lower=0).fillna(0.0)


def _pente(series: pd.Series, fenetre: int) -> pd.Series:
    """
    Pente (unités/heure) : écart entre la moyenne de la 2nde moitié et celle de la
    1ère moitié de la fenêtre, divisé par la durée d'une demi-fenêtre — proxy de
    tendance rapide (deux moyennes glissantes) plutôt qu'une régression OLS par
    fenêtre glissante (coûteuse sur 1,75M lignes). Détecte une accélération ou un
    ralentissement de l'encrassement/broutement, pas une estimation de pente exacte.
    """
    demi_fenetre = fenetre // 2
    moyenne_recente = series.rolling(demi_fenetre, min_periods=1).mean()
    moyenne_ancienne = series.shift(demi_fenetre).rolling(demi_fenetre, min_periods=1).mean()
    return ((moyenne_recente - moyenne_ancienne) / (demi_fenetre * 0.5)).fillna(0.0)


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
    # Fenêtres 72h/120h : lissent le bruit d'activité pas-à-pas pour voir la tendance
    # de fond du broutement (chatter, rampe terminale TWF, point 8) plutôt que ses
    # pics instantanés déjà couverts par vibration_max_24h.
    df["vibration_moyenne_72h"] = grp["vibration"].transform(lambda x: x.rolling(FENETRE_72H, min_periods=1).mean())
    df["vibration_ecart_type_72h"] = grp["vibration"].transform(lambda x: x.rolling(FENETRE_72H, min_periods=1).std().fillna(0.0))
    df["vibration_moyenne_120h"] = grp["vibration"].transform(lambda x: x.rolling(FENETRE_120H, min_periods=1).mean())

    df["temperature_process_max_24h"] = grp["temperature_process"].transform(
        lambda x: x.rolling(FENETRE_24H, min_periods=1).max()
    )
    df["temperature_process_moyenne_24h"] = grp["temperature_process"].transform(
        lambda x: x.rolling(FENETRE_24H, min_periods=1).mean()
    )
    df["temperature_process_delta_4h"] = grp["temperature_process"].transform(lambda x: _delta(x, FENETRE_4H))
    df["temperature_process_moyenne_72h"] = grp["temperature_process"].transform(
        lambda x: x.rolling(FENETRE_72H, min_periods=1).mean()
    )
    df["temperature_process_moyenne_120h"] = grp["temperature_process"].transform(
        lambda x: x.rolling(FENETRE_120H, min_periods=1).mean()
    )

    # Écart thermique process/ambiant : le générateur définit HDF sur cette grandeur
    # (moyennée 4h), mais elle n'était pas exposée comme feature au modèle — gain
    # quasi gratuit sur le mode le moins bien détecté (point 19).
    df["ecart_thermique"] = df["temperature_process"] - df["temperature_ambiante"]
    # Pic 24h de l'écart thermique — HDF vérifié sous-détecté (68 % de recall) avec
    # seulement la valeur instantanée et les agrégats sur temperature_process seul ;
    # le pic sur ecart_thermique lui-même colle davantage au mécanisme réel du générateur
    # (surchauffe soutenue). Gain mesuré : +7 pts de recall HDF. Cf. docs/model-decisions.md.
    df["ecart_thermique_max_24h"] = grp["ecart_thermique"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).max())
    # Moyenne 72h/120h : la chaîne encrassement → dissipation réduite → HDF (cf.
    # docs/data-generation.md) est un phénomène qui s'installe sur plusieurs jours,
    # pas seulement un pic sur 24h — cherche le signal "surchauffe soutenue" à plus
    # long terme que ecart_thermique_max_24h.
    df["ecart_thermique_moyenne_72h"] = grp["ecart_thermique"].transform(lambda x: x.rolling(FENETRE_72H, min_periods=1).mean())
    df["ecart_thermique_moyenne_120h"] = grp["ecart_thermique"].transform(lambda x: x.rolling(FENETRE_120H, min_periods=1).mean())
    # Pente 72h : vitesse d'accumulation de l'écart thermique — un ecart_thermique
    # élevé mais stable n'est pas le même signal qu'un ecart_thermique qui monte
    # rapidement (encrassement en cours d'installation).
    df["pente_ecart_thermique_72h"] = grp["ecart_thermique"].transform(lambda x: _pente(x, FENETRE_72H))
    # Pente 72h de la vibration : vitesse d'accumulation du broutement (chatter),
    # même logique que ci-dessus appliquée à la rampe terminale TWF (point 8).
    df["pente_vibration_72h"] = grp["vibration"].transform(lambda x: _pente(x, FENETRE_72H))

    # Poussière cumulée sur 24h : signal précurseur de l'encrassement (chaîne
    # poussière → HDF, cf. docs/data-generation.md), pas juste la lecture instantanée.
    df["poussiere_moyenne_24h"] = grp["concentration_poussiere"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).mean())
    df["poussiere_max_24h"] = grp["concentration_poussiere"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).max())
    df["poussiere_moyenne_72h"] = grp["concentration_poussiere"].transform(lambda x: x.rolling(FENETRE_72H, min_periods=1).mean())
    df["poussiere_moyenne_120h"] = grp["concentration_poussiere"].transform(
        lambda x: x.rolling(FENETRE_120H, min_periods=1).mean()
    )

    # Pic de couple sur 24h : le couple instantané est bruyant (varie avec l'activité
    # à chaque pas) et noyait le signal précurseur OSF (couple élevé combiné à
    # l'usure) — vérifié sous-détecté (47 % de recall) avec le couple brut seul.
    # Gain mesuré : +10 pts de recall OSF. Cf. docs/model-decisions.md.
    df["couple_max_24h"] = grp["couple"].transform(lambda x: x.rolling(FENETRE_24H, min_periods=1).max())

    # Ratio vibration/couple : isole l'usure mécanique (roulements, broutement) des
    # simples variations de charge de production — une vibration élevée SOUS charge
    # élevée est normal (production intense), une vibration élevée SANS charge
    # correspondante est un signal d'usure. Plancher à 1 Nm pour éviter l'explosion
    # du ratio quand le couple approche 0 (machine à l'arrêt).
    df["ratio_vibration_couple"] = df["vibration"] / df["couple"].clip(lower=1.0)

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
    df["vibration_moyenne_72h"] = df["vibration"].rolling(FENETRE_72H, min_periods=1).mean()
    df["vibration_ecart_type_72h"] = df["vibration"].rolling(FENETRE_72H, min_periods=1).std().fillna(0.0)
    df["vibration_moyenne_120h"] = df["vibration"].rolling(FENETRE_120H, min_periods=1).mean()

    df["temperature_process_max_24h"] = df["temperature_process"].rolling(FENETRE_24H, min_periods=1).max()
    df["temperature_process_moyenne_24h"] = df["temperature_process"].rolling(FENETRE_24H, min_periods=1).mean()
    df["temperature_process_delta_4h"] = _delta(df["temperature_process"], FENETRE_4H)
    df["temperature_process_moyenne_72h"] = df["temperature_process"].rolling(FENETRE_72H, min_periods=1).mean()
    df["temperature_process_moyenne_120h"] = df["temperature_process"].rolling(FENETRE_120H, min_periods=1).mean()

    df["ecart_thermique"] = df["temperature_process"] - df["temperature_ambiante"]
    df["ecart_thermique_max_24h"] = df["ecart_thermique"].rolling(FENETRE_24H, min_periods=1).max()
    df["ecart_thermique_moyenne_72h"] = df["ecart_thermique"].rolling(FENETRE_72H, min_periods=1).mean()
    df["ecart_thermique_moyenne_120h"] = df["ecart_thermique"].rolling(FENETRE_120H, min_periods=1).mean()
    df["pente_ecart_thermique_72h"] = _pente(df["ecart_thermique"], FENETRE_72H)
    df["pente_vibration_72h"] = _pente(df["vibration"], FENETRE_72H)

    df["poussiere_moyenne_24h"] = df["concentration_poussiere"].rolling(FENETRE_24H, min_periods=1).mean()
    df["poussiere_max_24h"] = df["concentration_poussiere"].rolling(FENETRE_24H, min_periods=1).max()
    df["poussiere_moyenne_72h"] = df["concentration_poussiere"].rolling(FENETRE_72H, min_periods=1).mean()
    df["poussiere_moyenne_120h"] = df["concentration_poussiere"].rolling(FENETRE_120H, min_periods=1).mean()

    df["couple_max_24h"] = df["couple"].rolling(FENETRE_24H, min_periods=1).max()
    df["ratio_vibration_couple"] = df["vibration"] / df["couple"].clip(lower=1.0)
    return df


def _pente_historique(valeurs: list[float], fenetre: int) -> float:
    """Version liste de `_pente`, pour le chemin API (historique + valeur courante)."""
    demi_fenetre = max(1, min(fenetre // 2, len(valeurs) // 2))
    if len(valeurs) < 2:
        return 0.0
    moyenne_recente = float(np.mean(valeurs[-demi_fenetre:]))
    moyenne_ancienne = float(np.mean(valeurs[:demi_fenetre]))
    return (moyenne_recente - moyenne_ancienne) / (demi_fenetre * 0.5)


def enrichir_point_inference(current: dict, history_df: pd.DataFrame) -> dict:
    """
    Calcule les features temporelles pour un point unique (chemin API).
    current       : dict avec les features courantes (les 15 features brutes)
    history_df    : DataFrame des 240 dernières lignes de la machine (120h, ASC :
                     usure_outil, vibration, temperature_process, concentration_poussiere,
                     couple, temperature_ambiante), peut être vide.
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
    result["ratio_vibration_couple"] = vibration_actuelle / max(couple_actuel, 1.0)

    if history_df.empty:
        result["usure_outil_delta_24h"] = 0.0
        result["usure_outil_delta_4h"] = 0.0
        result["vibration_max_24h"] = vibration_actuelle
        result["vibration_moyenne_24h"] = vibration_actuelle
        result["vibration_ecart_type_24h"] = 0.0
        result["vibration_delta_4h"] = 0.0
        result["vibration_moyenne_72h"] = vibration_actuelle
        result["vibration_ecart_type_72h"] = 0.0
        result["vibration_moyenne_120h"] = vibration_actuelle
        result["temperature_process_max_24h"] = temperature_actuelle
        result["temperature_process_moyenne_24h"] = temperature_actuelle
        result["temperature_process_delta_4h"] = 0.0
        result["temperature_process_moyenne_72h"] = temperature_actuelle
        result["temperature_process_moyenne_120h"] = temperature_actuelle
        result["ecart_thermique_max_24h"] = ecart_thermique_actuel
        result["ecart_thermique_moyenne_72h"] = ecart_thermique_actuel
        result["ecart_thermique_moyenne_120h"] = ecart_thermique_actuel
        result["pente_ecart_thermique_72h"] = 0.0
        result["pente_vibration_72h"] = 0.0
        result["poussiere_moyenne_24h"] = poussiere_actuelle
        result["poussiere_max_24h"] = poussiere_actuelle
        result["poussiere_moyenne_72h"] = poussiere_actuelle
        result["poussiere_moyenne_120h"] = poussiere_actuelle
        result["couple_max_24h"] = couple_actuel
        return result

    def _delta_historique(col: str, current_value: float, window: int) -> float:
        hist = history_df[col].dropna()
        if len(hist) < window:
            return 0.0
        return max(0.0, current_value - float(hist.iloc[-window]))

    def _stats_fenetres(historique: list[float], valeur_actuelle: float) -> dict[str, float]:
        """Moyenne (+écart-type) sur les 3 fenêtres, calculées sur la portion d'historique
        réellement disponible (jusqu'à 240 lignes/120h) — dégrade proprement avec moins
        d'historique plutôt que de planter."""
        serie = historique + [valeur_actuelle]
        return {
            "24h_moyenne": float(np.mean(serie[-FENETRE_24H:])),
            "24h_ecart_type": float(np.std(serie[-FENETRE_24H:])) if len(serie[-FENETRE_24H:]) > 1 else 0.0,
            "24h_max": float(np.max(serie[-FENETRE_24H:])),
            "72h_moyenne": float(np.mean(serie[-FENETRE_72H:])),
            "72h_ecart_type": float(np.std(serie[-FENETRE_72H:])) if len(serie[-FENETRE_72H:]) > 1 else 0.0,
            "120h_moyenne": float(np.mean(serie[-FENETRE_120H:])),
        }

    # Usure : delta 24h (24e ligne en partant de la fin, ou la plus ancienne
    # disponible si l'historique est plus court que 24h — désormais possible car
    # history_df peut contenir jusqu'à 240 lignes/120h, pas seulement 48/24h) et 4h.
    result["usure_outil_delta_24h"] = _delta_historique("usure_outil", usure_actuelle, FENETRE_24H)
    result["usure_outil_delta_4h"] = _delta_historique("usure_outil", usure_actuelle, FENETRE_4H)

    # Vibration : max/moyenne/écart-type sur 24h/72h/120h + valeur courante, delta 4h,
    # pente 72h (vitesse d'accumulation du broutement, point 8).
    historique_vibration = history_df["vibration"].dropna().tolist()
    stats_vibration = _stats_fenetres(historique_vibration, vibration_actuelle)
    result["vibration_max_24h"] = stats_vibration["24h_max"]
    result["vibration_moyenne_24h"] = stats_vibration["24h_moyenne"]
    result["vibration_ecart_type_24h"] = stats_vibration["24h_ecart_type"]
    result["vibration_delta_4h"] = _delta_historique("vibration", vibration_actuelle, FENETRE_4H)
    result["vibration_moyenne_72h"] = stats_vibration["72h_moyenne"]
    result["vibration_ecart_type_72h"] = stats_vibration["72h_ecart_type"]
    result["vibration_moyenne_120h"] = stats_vibration["120h_moyenne"]
    result["pente_vibration_72h"] = _pente_historique(historique_vibration[-FENETRE_72H:] + [vibration_actuelle], FENETRE_72H)

    # Température process : max/moyenne sur 24h/72h/120h + valeur courante, delta 4h
    historique_temperature = history_df["temperature_process"].dropna().tolist()
    stats_temperature = _stats_fenetres(historique_temperature, temperature_actuelle)
    result["temperature_process_max_24h"] = stats_temperature["24h_max"]
    result["temperature_process_moyenne_24h"] = stats_temperature["24h_moyenne"]
    result["temperature_process_delta_4h"] = _delta_historique("temperature_process", temperature_actuelle, FENETRE_4H)
    result["temperature_process_moyenne_72h"] = stats_temperature["72h_moyenne"]
    result["temperature_process_moyenne_120h"] = stats_temperature["120h_moyenne"]

    # Poussière : max/moyenne sur 24h/72h/120h + valeur courante
    historique_poussiere = (
        history_df["concentration_poussiere"].dropna().tolist() if "concentration_poussiere" in history_df else []
    )
    stats_poussiere = _stats_fenetres(historique_poussiere, poussiere_actuelle)
    result["poussiere_moyenne_24h"] = stats_poussiere["24h_moyenne"]
    result["poussiere_max_24h"] = stats_poussiere["24h_max"]
    result["poussiere_moyenne_72h"] = stats_poussiere["72h_moyenne"]
    result["poussiere_moyenne_120h"] = stats_poussiere["120h_moyenne"]

    # Couple : pic sur historique + valeur courante (signal précurseur OSF, point
    # "amélioration détection HDF/OSF" — cf. docs/model-decisions.md).
    historique_couple = history_df["couple"].dropna().tolist() + [couple_actuel] if "couple" in history_df else [couple_actuel]
    result["couple_max_24h"] = float(np.max(historique_couple[-FENETRE_24H:]))

    # Écart thermique : pic 24h, moyennes 72h/120h + valeur courante, pente 72h
    # (vitesse d'installation de l'encrassement, cf. docs/data-generation.md).
    if "temperature_process" in history_df and "temperature_ambiante" in history_df:
        historique_ambiante = history_df["temperature_ambiante"].dropna()
        historique_process = history_df["temperature_process"].dropna()
        historique_ecart_thermique = (historique_process - historique_ambiante).dropna().tolist()
    else:
        historique_ecart_thermique = []
    stats_ecart_thermique = _stats_fenetres(historique_ecart_thermique, ecart_thermique_actuel)
    result["ecart_thermique_max_24h"] = stats_ecart_thermique["24h_max"]
    result["ecart_thermique_moyenne_72h"] = stats_ecart_thermique["72h_moyenne"]
    result["ecart_thermique_moyenne_120h"] = stats_ecart_thermique["120h_moyenne"]
    result["pente_ecart_thermique_72h"] = _pente_historique(
        historique_ecart_thermique[-FENETRE_72H:] + [ecart_thermique_actuel], FENETRE_72H
    )

    return result
