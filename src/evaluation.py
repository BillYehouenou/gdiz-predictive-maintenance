"""
Métrique événementielle, celle qui compte pour l'atelier :
- recall événementiel : part des pannes précédées d'au moins une alerte dans l'horizon ;
- fausses alertes par machine-jour, comptées en épisodes (une alerte continue = une intervention).
"""

import numpy as np
import pandas as pd

from src.config import PAS
from src.donnees import MODES_PANNE_PREDICTIBLES

# Un épisode d'alerte plus long que 24h est redécoupé : sinon un modèle bloqué en alerte compterait pour un seul épisode
PAS_MAX_EPISODE_DEFAUT = 48


def _episodes_alerte(alertes: np.ndarray, pas_max_episode: int | None = PAS_MAX_EPISODE_DEFAUT) -> list[tuple[int, int]]:
    """(début, fin) des suites consécutives d'alertes, redécoupées au-delà de pas_max_episode (None : jamais)."""
    if alertes.sum() == 0:
        return []
    diff = np.diff(np.concatenate([[0], alertes.astype(int), [0]]))
    episodes = list(zip(np.flatnonzero(diff == 1).tolist(), (np.flatnonzero(diff == -1) - 1).tolist()))
    if pas_max_episode is None:
        return episodes
    return [(d, min(d + pas_max_episode - 1, fin)) for debut, fin in episodes for d in range(debut, fin + 1, pas_max_episode)]


def metriques_evenementielles(
    df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float,
    pas_horizon: int,
    modes_panne: tuple = MODES_PANNE_PREDICTIBLES,
    pas_max_episode: int | None = PAS_MAX_EPISODE_DEFAUT,
) -> dict:
    """
    df : id_machine, timestamp, panne, mode_panne, aligné ligne à ligne avec probas.
    Une panne est attrapée si un épisode d'alerte chevauche [panne - horizon, panne] ; le préavis court
    depuis le début de cet épisode. Les épisodes associés à aucune panne sont des fausses alertes.
    """
    if len(df) != len(probas):
        raise ValueError(f"df et probas doivent avoir la même longueur ({len(df)} vs {len(probas)}).")

    df = df.reset_index(drop=True)
    alertes = np.asarray(probas) >= threshold
    nb_pannes = nb_attrapees = nb_episodes = nb_fausses_alertes = 0
    preavis_heures: list[float] = []
    machine_jours = 0.0

    for _, g in df.groupby("id_machine", sort=False):
        g = g.sort_values("timestamp")
        machine_jours += (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]) / pd.Timedelta("1D")
        episodes = _episodes_alerte(alertes[g.index.to_numpy()], pas_max_episode)
        nb_episodes += len(episodes)

        pannes = np.flatnonzero(((g["panne"] == 1) & g["mode_panne"].isin(modes_panne)).to_numpy())
        associes = set()
        for f in pannes.tolist():
            nb_pannes += 1
            candidats = [i for i, (debut, fin) in enumerate(episodes) if debut <= f and fin >= f - pas_horizon]
            if candidats:
                nb_attrapees += 1
                dernier = max(candidats, key=lambda i: episodes[i][0])
                associes.add(dernier)
                preavis_heures.append((f - episodes[dernier][0]) * PAS / pd.Timedelta("1h"))
        nb_fausses_alertes += len(episodes) - len(associes)

    return {
        "recall_evenementiel": nb_attrapees / nb_pannes if nb_pannes else 0.0,
        "nb_pannes": nb_pannes,
        "nb_attrapees": nb_attrapees,
        "nb_episodes": nb_episodes,
        "nb_episodes_fausses_alertes": nb_fausses_alertes,
        "taux_fausses_alertes_par_machine_jour": nb_fausses_alertes / machine_jours if machine_jours else 0.0,
        "preavis_heures": preavis_heures,
    }


def grille_seuils(probas_validation: np.ndarray) -> np.ndarray:
    """Seuils candidats sur les quantiles des scores, de la médiane au 99,9e centile, plus serrés vers le haut."""
    return np.unique(np.quantile(probas_validation, 1 - np.geomspace(0.5, 0.001, 200)))


def trouver_seuil_pour_budget_fa(
    df: pd.DataFrame,
    probas: np.ndarray,
    thresholds: np.ndarray,
    pas_horizon: int,
    budget_par_machine_jour: float,
    modes_panne: tuple = MODES_PANNE_PREDICTIBLES,
    pas_max_episode: int | None = PAS_MAX_EPISODE_DEFAUT,
) -> tuple[float, dict]:
    """Seuil qui maximise le recall sous le budget de fausses alertes ; à défaut, celui qui en fait le moins."""
    resultats = [
        (float(t), metriques_evenementielles(df, probas, float(t), pas_horizon, modes_panne, pas_max_episode)) for t in thresholds
    ]
    dans_budget = [(t, m) for t, m in resultats if m["taux_fausses_alertes_par_machine_jour"] <= budget_par_machine_jour]
    if dans_budget:
        return max(dans_budget, key=lambda tm: tm[1]["recall_evenementiel"])
    return min(resultats, key=lambda tm: tm[1]["taux_fausses_alertes_par_machine_jour"])
