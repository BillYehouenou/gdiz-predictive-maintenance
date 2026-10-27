"""
Mesures qui comptent pour l'atelier : part des pannes annoncées dans l'horizon (et avec au moins 4 h de préavis),
fausses alertes par machine-jour comptées en épisodes (une alerte continue = une intervention), précision des alertes
et bilan en FCFA. Une machine qui ne tourne pas ne déclenche jamais d'alerte.
"""

import numpy as np
import pandas as pd

from src.config import PAS
from src.donnees import HORIZON, MODES_PANNE_PREDICTIBLES
from src.metier import COUT_INSPECTION, gain_panne_evitee

BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR = 0.2  # dérangements inutiles acceptés par l'équipe de maintenance
PAS_HORIZON = int(HORIZON / PAS)
PREAVIS_UTILE_H = 4.0  # en dessous, la panne est annoncée trop tard pour planifier quoi que ce soit
PAS_MAX_EPISODE_DEFAUT = 48  # une alerte de plus de 24 h compte pour plusieurs épisodes : sinon, alerter tout le temps paierait


def _ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def _episodes_alerte(alertes: np.ndarray, pas_max_episode: int | None = PAS_MAX_EPISODE_DEFAUT) -> list[tuple[int, int]]:
    """(début, fin) des suites consécutives d'alertes, redécoupées au-delà de pas_max_episode (None : jamais)."""
    if alertes.sum() == 0:
        return []
    diff = np.diff(np.concatenate([[0], alertes.astype(int), [0]]))
    episodes = list(zip(np.flatnonzero(diff == 1).tolist(), (np.flatnonzero(diff == -1) - 1).tolist()))
    if pas_max_episode is None:
        return episodes
    return [(d, min(d + pas_max_episode - 1, fin)) for debut, fin in episodes for d in range(debut, fin + 1, pas_max_episode)]


def metriques_evenementielles(df: pd.DataFrame, probas: np.ndarray, threshold: float, pas_horizon: int,
                              modes_panne: tuple = MODES_PANNE_PREDICTIBLES,
                              pas_max_episode: int | None = PAS_MAX_EPISODE_DEFAUT) -> dict:  # fmt: skip
    """
    df : id_machine, timestamp, panne, mode_panne (et etat), aligné avec probas. Une panne est attrapée si un épisode
    d'alerte chevauche [panne - horizon, panne] ; tous ces épisodes lui reviennent et le préavis court depuis le début
    du premier, borné à l'horizon. Les épisodes associés à aucune panne sont des fausses alertes.
    """
    if len(df) != len(probas):
        raise ValueError(f"df et probas doivent avoir la même longueur ({len(df)} vs {len(probas)}).")
    df = df.reset_index(drop=True)
    alertes = np.asarray(probas) >= threshold
    if "etat" in df:
        alertes &= (df["etat"] == "marche").to_numpy()
    nb_pannes = nb_episodes = nb_fausses = 0
    preavis: list[float] = []
    attrapees: list[int] = []  # lignes des pannes attrapées, pour le bilan économique
    machine_jours = 0.0
    for _, g in df.groupby("id_machine", sort=False):
        g = g.sort_values("timestamp")
        machine_jours += (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]) / pd.Timedelta("1D")
        episodes = _episodes_alerte(alertes[g.index.to_numpy()], pas_max_episode)
        associes = set()
        for f in np.flatnonzero(((g["panne"] == 1) & g["mode_panne"].isin(modes_panne)).to_numpy()).tolist():
            nb_pannes += 1
            candidats = [i for i, (debut, fin) in enumerate(episodes) if debut <= f and fin >= f - pas_horizon]
            if candidats:
                associes.update(candidats)
                premier = min(episodes[i][0] for i in candidats)
                preavis.append(min(f - premier, pas_horizon) * PAS / pd.Timedelta("1h"))
                attrapees.append(int(g.index[f]))
        nb_episodes += len(episodes)
        nb_fausses += len(episodes) - len(associes)
    nb_attrapees = len(attrapees)
    return {
        "recall_evenementiel": _ratio(nb_attrapees, nb_pannes),
        "recall_preavis_4h": _ratio(sum(h >= PREAVIS_UTILE_H for h in preavis), nb_pannes),
        "precision_alertes": _ratio(nb_episodes - nb_fausses, nb_episodes),  # part des épisodes utiles
        "nb_pannes": nb_pannes,
        "nb_attrapees": nb_attrapees,
        "nb_episodes": nb_episodes,
        "nb_episodes_fausses_alertes": nb_fausses,
        "taux_fausses_alertes_par_machine_jour": _ratio(nb_fausses, machine_jours),
        "preavis_heures": preavis,
        "attrapees": attrapees,
    }


def duree_reparation_h(df: pd.DataFrame, ligne: int) -> float:
    """Heures passées en état « panne » après la ligne de panne (comme metier.evenements), au moins un pas."""
    etats, machine = df["etat"].to_numpy(), df["id_machine"].to_numpy()
    fin = ligne + 1
    while fin < len(df) and machine[fin] == machine[ligne] and etats[fin] == "panne":
        fin += 1
    return max(fin - ligne - 1, 1) * PAS / pd.Timedelta("1h")


def bilan_economique(df: pd.DataFrame, metriques: dict) -> dict:
    """Valeur des pannes évitées (durée réelle de chaque réparation), gain net, et coût d'inspection au-delà duquel on perd."""
    df = df.reset_index(drop=True)
    valeur = sum(gain_panne_evitee(df["type_machine"].iat[i], duree_reparation_h(df, i)) for i in metriques["attrapees"])
    fausses = metriques["nb_episodes_fausses_alertes"]
    return {"valeur_pannes_evitees": float(valeur), "gain_net": float(valeur - fausses * COUT_INSPECTION),
            "cout_inspection_rentable": float(valeur / fausses) if fausses else float("inf")}  # fmt: skip


def grille_seuils(probas_validation: np.ndarray) -> np.ndarray:
    """Seuils candidats sur les quantiles des scores, de la médiane au 99,9e centile, plus serrés vers le haut."""
    return np.unique(np.quantile(probas_validation, 1 - np.geomspace(0.5, 0.001, 200)))


def trouver_seuil_pour_budget_fa(df: pd.DataFrame, probas: np.ndarray, thresholds: np.ndarray, pas_horizon: int,
                                 budget_par_machine_jour: float, modes_panne: tuple = MODES_PANNE_PREDICTIBLES,
                                 pas_max_episode: int | None = PAS_MAX_EPISODE_DEFAUT) -> tuple[float, dict]:  # fmt: skip
    """Seuil qui maximise le recall sous le budget de fausses alertes ; à défaut, celui qui en fait le moins."""
    resultats = [(float(t), metriques_evenementielles(df, probas, float(t), pas_horizon, modes_panne, pas_max_episode))
                 for t in thresholds]  # fmt: skip
    dans_budget = [tm for tm in resultats if tm[1]["taux_fausses_alertes_par_machine_jour"] <= budget_par_machine_jour]
    if dans_budget:
        return max(dans_budget, key=lambda tm: tm[1]["recall_evenementiel"])
    return min(resultats, key=lambda tm: tm[1]["taux_fausses_alertes_par_machine_jour"])
