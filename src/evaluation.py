"""
Métrique événementielle : remplace le F2 par-pas de temps comme boussole du projet.

Deux chiffres qui comptent, pas un score composite abstrait :
- recall événementiel : part des pannes réelles précédées d'au moins une alerte
  dans les X pas qui précèdent.
- taux de fausses alertes par machine-jour, compté en épisodes (une alerte continue
  sur une même machine = un seul événement, correspondant à une intervention).
"""

import numpy as np
import pandas as pd

from src.features import MODES_PANNE_PREDICTIBLES

# Délai de réarmement par défaut (24h = 48 pas de 30 min) : au-delà de cette durée,
# un épisode d'alerte continu est redécoupé en plusieurs épisodes. Sans ça, un modèle 
# qui reste bloqué en alerte pendant des mois ne compte jamais que comme UN épisode.
PAS_MAX_EPISODE_DEFAUT = 48


def _episodes_alerte(
    indicateurs_alerte: np.ndarray, pas_max_episode: int | None = PAS_MAX_EPISODE_DEFAUT
) -> list[tuple[int, int]]:
    """
    Runs consécutifs à 1 dans un vecteur binaire trié par temps.
    pas_max_episode : redécoupe tout run plus long que cette durée.
    None : désactive le redécoupage.
    """
    if indicateurs_alerte.sum() == 0:
        return []
    padded = np.concatenate([[0], indicateurs_alerte.astype(int), [0]])
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    episodes = list(zip(starts.tolist(), ends.tolist()))

    if pas_max_episode is None:
        return episodes

    split_episodes = []
    for s, e in episodes:
        cur = s
        while cur <= e:
            seg_end = min(cur + pas_max_episode - 1, e)
            split_episodes.append((cur, seg_end))
            cur = seg_end + 1
    return split_episodes


def metriques_evenementielles(
    df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float,
    pas_horizon: int,
    modes_panne: tuple = MODES_PANNE_PREDICTIBLES,
    pas_max_episode: int | None = PAS_MAX_EPISODE_DEFAUT,
) -> dict:
    """
    df : doit contenir id_machine, timestamp, panne, mode_panne, aligné index à index avec probas.

    Pour chaque panne réelle, cherche le dernier épisode d'alerte qui chevauche la
    fenêtre [panne - pas_horizon, panne] sur la même machine : s'il existe, la
    panne est "attrapée" et le préavis = temps entre le début de cet épisode et la
    panne. Les épisodes non associés à une panne sont des fausses alertes.

    pas_max_episode : délai de réarmement - un épisode continu plus long est redécoupé, 
    pour qu'un modèle en alerte quasi permanente ne soit pas artificiellement bien noté.
    """
    if len(df) != len(probas):
        raise ValueError(f"df et probas doivent avoir la même longueur ({len(df)} vs {len(probas)}).")

    df = df.reset_index(drop=True)
    indicateurs_alerte_tous = (np.asarray(probas) >= threshold).astype(int)

    total_pannes = 0
    total_attrapees = 0
    total_episodes = 0
    total_episodes_fausses_alertes = 0
    preavis_heures: list[float] = []
    total_machine_jours = 0.0

    for _, g in df.groupby("id_machine", sort=False):
        g = g.sort_values("timestamp")
        positions = g.index.to_numpy()
        g = g.reset_index(drop=True)
        a = indicateurs_alerte_tous[positions]

        duree_heures_totale = (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds() / 3600.0
        total_machine_jours += duree_heures_totale / 24.0

        episodes = _episodes_alerte(a, pas_max_episode=pas_max_episode)
        total_episodes += len(episodes)

        is_failure = (g["panne"] == 1) & g["mode_panne"].isin(modes_panne)
        index_pannes = np.flatnonzero(is_failure.to_numpy())

        index_episodes_associes = set()
        for f_raw in index_pannes:
            f = int(f_raw)
            total_pannes += 1
            debut_fenetre = max(0, f - pas_horizon)
            candidates = [i for i, (s, e) in enumerate(episodes) if s <= f and e >= debut_fenetre]
            if candidates:
                total_attrapees += 1
                best = max(candidates, key=lambda i: episodes[i][0])
                index_episodes_associes.add(best)
                preavis_heures.append((f - episodes[best][0]) * 0.5)

        total_episodes_fausses_alertes += len(episodes) - len(index_episodes_associes)

    return {
        "recall_evenementiel": total_attrapees / total_pannes if total_pannes > 0 else 0.0,
        "nb_pannes": total_pannes,
        "nb_attrapees": total_attrapees,
        "nb_episodes": total_episodes,
        "nb_episodes_fausses_alertes": total_episodes_fausses_alertes,
        "taux_fausses_alertes_par_machine_jour": (
            total_episodes_fausses_alertes / total_machine_jours if total_machine_jours > 0 else 0.0
        ),
        "preavis_heures": preavis_heures,
    }


def trouver_seuil_pour_budget_fa(
    df: pd.DataFrame,
    probas: np.ndarray,
    thresholds: np.ndarray,
    pas_horizon: int,
    budget_par_machine_jour: float,
    modes_panne: tuple = MODES_PANNE_PREDICTIBLES,
    pas_max_episode: int | None = PAS_MAX_EPISODE_DEFAUT,
) -> tuple[float, dict]:
    """
    Balaie une grille de seuils, ne retient que ceux dont le taux de fausses alertes
    respecte le budget, choisit celui qui maximise le recall événementiel parmi eux.
    Si aucun seuil ne respecte le budget (modèle trop bruité ou budget trop strict),
    retombe sur celui qui minimise le taux de fausses alertes plutôt que de planter
    ou de retourner un seuil arbitraire.
    """
    results = [
        (float(t), metriques_evenementielles(df, probas, float(t), pas_horizon, modes_panne, pas_max_episode)) for t in thresholds
    ]

    dans_budget = [(t, m) for t, m in results if m["taux_fausses_alertes_par_machine_jour"] <= budget_par_machine_jour]
    if dans_budget:
        return max(dans_budget, key=lambda tm: tm[1]["recall_evenementiel"])
    return min(results, key=lambda tm: tm[1]["taux_fausses_alertes_par_machine_jour"])
