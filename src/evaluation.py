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


def _alert_episodes(alert_flags: np.ndarray) -> list[tuple[int, int]]:
    """Runs consécutifs à 1 dans un vecteur binaire trié par temps — (start_idx, end_idx) inclusifs."""
    if alert_flags.sum() == 0:
        return []
    padded = np.concatenate([[0], alert_flags.astype(int), [0]])
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def event_based_metrics(
    df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float,
    horizon_steps: int,
    failure_modes: tuple = PREDICTABLE_FAILURE_MODES,
) -> dict:
    """
    df : doit contenir machine_id, timestamp, target, failure_mode, aligné index à
    index avec `probas` (même longueur, même ordre — pas besoin d'être pré-trié).

    Pour chaque panne réelle, cherche le dernier épisode d'alerte qui chevauche la
    fenêtre [panne - horizon_steps, panne] sur la même machine : s'il existe, la
    panne est "attrapée" et le préavis = temps entre le début de cet épisode et la
    panne. Les épisodes non associés à une panne sont des fausses alertes.
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

        episodes = _alert_episodes(a)
        total_episodes += len(episodes)

        is_failure = (g["target"] == 1) & g["failure_mode"].isin(failure_modes)
        failure_idxs = np.flatnonzero(is_failure.to_numpy())

        matched_episode_idx = set()
        for f_raw in failure_idxs:
            f = int(f_raw)
            total_failures += 1
            window_start = max(0, f - horizon_steps)
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
    horizon_steps: int,
    budget_per_machine_day: float,
    failure_modes: tuple = PREDICTABLE_FAILURE_MODES,
) -> tuple[float, dict]:
    """
    Balaie une grille de seuils, ne retient que ceux dont le taux de fausses alertes
    respecte le budget, choisit celui qui maximise le recall événementiel parmi eux.
    Si aucun seuil ne respecte le budget (modèle trop bruité ou budget trop strict),
    retombe sur celui qui minimise le taux de fausses alertes plutôt que de planter
    ou de retourner un seuil arbitraire.
    """
    results = [(float(t), event_based_metrics(df, probas, float(t), horizon_steps, failure_modes)) for t in thresholds]

    within_budget = [(t, m) for t, m in results if m["false_alarm_rate_per_machine_day"] <= budget_per_machine_day]
    if within_budget:
        return max(within_budget, key=lambda tm: tm[1]["recall_event"])
    return min(results, key=lambda tm: tm[1]["false_alarm_rate_per_machine_day"])
