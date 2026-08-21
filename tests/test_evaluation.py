"""
Tests de la métrique événementielle — logique isolée du reste du pipeline
d'entraînement, donc vérifiable par calcul manuel plutôt que par supposition.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import _alert_episodes, event_based_metrics, find_threshold_for_fa_budget


def test_alert_episodes_no_alerts():
    assert _alert_episodes(np.zeros(5, dtype=int)) == []


def test_alert_episodes_all_alerts():
    assert _alert_episodes(np.ones(5, dtype=int)) == [(0, 4)]


def test_alert_episodes_multiple_runs():
    a = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert _alert_episodes(a) == [(1, 2), (5, 5), (7, 9)]


def _make_single_machine_df(n: int, failure_idx: int, failure_mode: str = "TWF") -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=n, freq="30min")
    target = np.zeros(n, dtype=int)
    mode = np.full(n, "", dtype=object)
    target[failure_idx] = 1
    mode[failure_idx] = failure_mode
    return pd.DataFrame({"machine_id": ["M1"] * n, "timestamp": timestamps, "target": target, "failure_mode": mode})


def test_event_based_metrics_catches_failure_with_preceding_episode():
    """Épisode qui précède directement la panne -> attrapée ; épisode isolé plus tôt -> fausse alerte."""
    n = 20
    df = _make_single_machine_df(n, failure_idx=15)
    probas = np.zeros(n)
    probas[10:15] = 0.9  # épisode adjacent à la panne (idx 10-14)
    probas[0:3] = 0.9  # épisode isolé, hors fenêtre horizon_steps=10 -> [5, 15]

    metrics = event_based_metrics(df, probas, threshold=0.5, horizon_steps=10)

    assert metrics["n_failures"] == 1
    assert metrics["n_caught"] == 1
    assert metrics["recall_event"] == 1.0
    assert metrics["n_episodes"] == 2
    assert metrics["n_false_alarm_episodes"] == 1
    assert metrics["lead_times_hours"] == pytest.approx([2.5])


def test_event_based_metrics_misses_failure_without_episode():
    n = 20
    df = _make_single_machine_df(n, failure_idx=15)
    probas = np.zeros(n)

    metrics = event_based_metrics(df, probas, threshold=0.5, horizon_steps=10)

    assert metrics["n_failures"] == 1
    assert metrics["n_caught"] == 0
    assert metrics["recall_event"] == 0.0
    assert metrics["n_episodes"] == 0


def test_event_based_metrics_false_alarm_rate_per_machine_day():
    n = 48 * 2  # 2 jours pile, pas de 30 min
    df = pd.DataFrame(
        {
            "machine_id": ["M1"] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "target": np.zeros(n, dtype=int),
            "failure_mode": np.full(n, "", dtype=object),
        }
    )
    probas = np.zeros(n)
    probas[0:4] = 0.9  # 1 épisode de 2h, aucune panne -> 1 fausse alerte

    metrics = event_based_metrics(df, probas, threshold=0.5, horizon_steps=10)

    assert metrics["n_false_alarm_episodes"] == 1
    assert metrics["false_alarm_rate_per_machine_day"] == pytest.approx(0.5, rel=0.05)


def test_event_based_metrics_uses_per_mode_horizon_dict():
    """
    Point 15 : horizon_steps peut être un dict {mode: pas}. Une panne HDF avec un
    horizon court (3 pas) ne doit pas être attrapée par un épisode trop éloigné qui
    l'aurait été avec l'horizon TWF (plus long, 15 pas) — preuve que le bon horizon
    est bien appliqué par panne, pas un horizon unique pour tout le monde.
    """
    n = 40
    timestamps = pd.date_range("2025-01-01", periods=n, freq="30min")
    target = np.zeros(n, dtype=int)
    mode = np.full(n, "", dtype=object)
    target[10] = 1
    mode[10] = "HDF"
    target[30] = 1
    mode[30] = "TWF"
    df = pd.DataFrame({"machine_id": ["M1"] * n, "timestamp": timestamps, "target": target, "failure_mode": mode})

    probas = np.zeros(n)
    probas[2:5] = 0.9  # épisode A : hors fenêtre HDF (horizon 3 -> [7,10]), aurait chevauché avec horizon 15
    probas[20:25] = 0.9  # épisode B : dans la fenêtre TWF (horizon 15 -> [15,30])

    metrics = event_based_metrics(df, probas, threshold=0.5, horizon_steps={"HDF": 3, "TWF": 15})

    assert metrics["n_failures"] == 2
    assert metrics["n_caught"] == 1  # seule la panne TWF est attrapée
    assert metrics["recall_event"] == 0.5
    assert metrics["n_episodes"] == 2
    assert metrics["n_false_alarm_episodes"] == 1  # épisode A, hors fenêtre HDF


def test_find_threshold_for_fa_budget_falls_back_when_no_threshold_meets_budget():
    """Budget de 0 fausse alerte impossible à tenir dès qu'il y a du bruit -> repli, pas de crash."""
    n = 20
    df = _make_single_machine_df(n, failure_idx=15)
    probas = np.zeros(n)
    probas[10:15] = 0.9  # attrape la panne
    probas[0:2] = 0.9  # fausse alerte isolée, inévitable avec un seul seuil disponible

    best_t, best_m = find_threshold_for_fa_budget(
        df, probas, thresholds=np.array([0.5]), horizon_steps=10, budget_per_machine_day=0.0
    )

    assert best_t == 0.5
    assert best_m["false_alarm_rate_per_machine_day"] > 0.0


def test_find_threshold_for_fa_budget_maximizes_recall_within_budget():
    """Parmi les seuils respectant le budget, celui qui maximise le recall événementiel est choisi."""
    n = 20
    df = _make_single_machine_df(n, failure_idx=15)
    probas = np.zeros(n)
    probas[10:15] = 0.6  # épisode adjacent à la panne, alerte à seuil 0.5 mais pas 0.7

    best_t, best_m = find_threshold_for_fa_budget(
        df, probas, thresholds=np.array([0.5, 0.7]), horizon_steps=10, budget_per_machine_day=10.0
    )

    assert best_t == 0.5
    assert best_m["recall_event"] == 1.0
