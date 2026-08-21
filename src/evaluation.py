"""
Métrique événementielle : remplace le F2 par-pas de temps comme boussole du projet.

Deux chiffres qui comptent, pas un score composite abstrait :
- recall événementiel : part des pannes réelles précédées d'au moins une alerte
  dans les `horizon_steps` pas qui précèdent.
- taux de fausses alertes par machine-jour, compté en épisodes (une alerte continue
  sur une même machine = un seul événement, correspondant à une intervention), pas
  en pas de temps — un épisode de 10h ne pèse pas 20 "fausses alertes".
"""

import numpy as np
import pandas as pd

from src.features import PREDICTABLE_FAILURE_MODES

# Délai de réarmement par défaut (24h = 48 pas de 30 min) : au-delà de cette durée,
# un épisode d'alerte continu est redécoupé en plusieurs épisodes. Sans ça, un
# modèle qui reste bloqué en alerte pendant des mois ne compte jamais que comme UN
# épisode — angle mort vérifié empiriquement (cf. docs/model-decisions.md) qui
# permet à un modèle inutilisable (alerte quasi permanente) d'obtenir un taux de
# fausses alertes proche de zéro. 24h = un technicien qui n'a pas résolu l'alerte
# en une journée justifie un nouveau signalement, pas un silence indéfini.
DEFAULT_MAX_EPISODE_STEPS = 48


def _alert_episodes(alert_flags: np.ndarray, max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS) -> list[tuple[int, int]]:
    """
    Runs consécutifs à 1 dans un vecteur binaire trié par temps — (start_idx, end_idx)
    inclusifs. `max_episode_steps` : redécoupe tout run plus long que cette durée
    (délai de réarmement, cf. commentaire du module) ; None désactive le redécoupage.
    """
    if alert_flags.sum() == 0:
        return []
    padded = np.concatenate([[0], alert_flags.astype(int), [0]])
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    episodes = list(zip(starts.tolist(), ends.tolist()))

    if max_episode_steps is None:
        return episodes

    split_episodes = []
    for s, e in episodes:
        cur = s
        while cur <= e:
            seg_end = min(cur + max_episode_steps - 1, e)
            split_episodes.append((cur, seg_end))
            cur = seg_end + 1
    return split_episodes


def event_based_metrics(
    df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float,
    horizon_steps: int | dict[str, int],
    failure_modes: tuple = PREDICTABLE_FAILURE_MODES,
    max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
) -> dict:
    """
    df : doit contenir machine_id, timestamp, target, failure_mode, aligné index à
    index avec `probas` (même longueur, même ordre — pas besoin d'être pré-trié).

    Pour chaque panne réelle, cherche le dernier épisode d'alerte qui chevauche la
    fenêtre [panne - horizon_steps, panne] sur la même machine : s'il existe, la
    panne est "attrapée" et le préavis = temps entre le début de cet épisode et la
    panne. Les épisodes non associés à une panne sont des fausses alertes.

    `horizon_steps` : soit un entier unique (même fenêtre pour tous les modes), soit
    un dict {mode: pas} pour une fenêtre propre à chaque mode de panne (point 15 —
    HDF/OSF perdent en précision au-delà de 72h, TWF reste optimal à 120h).

    `max_episode_steps` : délai de réarmement (cf. DEFAULT_MAX_EPISODE_STEPS) — un
    épisode continu plus long est redécoupé, pour qu'un modèle en alerte quasi
    permanente ne soit pas artificiellement bien noté.
    """
    if len(df) != len(probas):
        raise ValueError(f"df et probas doivent avoir la même longueur ({len(df)} vs {len(probas)}).")

    df = df.reset_index(drop=True)
    alert_flags_all = (np.asarray(probas) >= threshold).astype(int)

    total_failures = 0
    total_caught = 0
    total_episodes = 0
    total_false_alarm_episodes = 0
    lead_times_hours: list[float] = []
    total_machine_days = 0.0

    for _, g in df.groupby("machine_id", sort=False):
        g = g.sort_values("timestamp")
        positions = g.index.to_numpy()
        g = g.reset_index(drop=True)
        a = alert_flags_all[positions]

        span_hours = (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds() / 3600.0
        total_machine_days += span_hours / 24.0

        episodes = _alert_episodes(a, max_episode_steps=max_episode_steps)
        total_episodes += len(episodes)

        is_failure = (g["target"] == 1) & g["failure_mode"].isin(failure_modes)
        failure_idxs = np.flatnonzero(is_failure.to_numpy())
        failure_modes_arr = g["failure_mode"].to_numpy()

        matched_episode_idx = set()
        for f_raw in failure_idxs:
            f = int(f_raw)
            total_failures += 1
            steps = horizon_steps[failure_modes_arr[f]] if isinstance(horizon_steps, dict) else horizon_steps
            window_start = max(0, f - steps)
            candidates = [i for i, (s, e) in enumerate(episodes) if s <= f and e >= window_start]
            if candidates:
                total_caught += 1
                best = max(candidates, key=lambda i: episodes[i][0])
                matched_episode_idx.add(best)
                lead_times_hours.append((f - episodes[best][0]) * 0.5)

        total_false_alarm_episodes += len(episodes) - len(matched_episode_idx)

    return {
        "recall_event": total_caught / total_failures if total_failures > 0 else 0.0,
        "n_failures": total_failures,
        "n_caught": total_caught,
        "n_episodes": total_episodes,
        "n_false_alarm_episodes": total_false_alarm_episodes,
        "false_alarm_rate_per_machine_day": (total_false_alarm_episodes / total_machine_days if total_machine_days > 0 else 0.0),
        "lead_times_hours": lead_times_hours,
    }


def find_threshold_for_fa_budget(
    df: pd.DataFrame,
    probas: np.ndarray,
    thresholds: np.ndarray,
    horizon_steps: int | dict[str, int],
    budget_per_machine_day: float,
    failure_modes: tuple = PREDICTABLE_FAILURE_MODES,
    max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
) -> tuple[float, dict]:
    """
    Balaie une grille de seuils, ne retient que ceux dont le taux de fausses alertes
    respecte le budget, choisit celui qui maximise le recall événementiel parmi eux.
    Si aucun seuil ne respecte le budget (modèle trop bruité ou budget trop strict),
    retombe sur celui qui minimise le taux de fausses alertes plutôt que de planter
    ou de retourner un seuil arbitraire.
    """
    results = [
        (float(t), event_based_metrics(df, probas, float(t), horizon_steps, failure_modes, max_episode_steps)) for t in thresholds
    ]

    within_budget = [(t, m) for t, m in results if m["false_alarm_rate_per_machine_day"] <= budget_per_machine_day]
    if within_budget:
        return max(within_budget, key=lambda tm: tm[1]["recall_event"])
    return min(results, key=lambda tm: tm[1]["false_alarm_rate_per_machine_day"])
