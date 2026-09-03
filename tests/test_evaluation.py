"""
Tests de la métrique événementielle : logique isolée du reste du pipeline
d'entraînement, donc vérifiable par calcul manuel plutôt que par supposition.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import _episodes_alerte, metriques_evenementielles, trouver_seuil_pour_budget_fa


def test_episodes_alerte_aucune_alerte():
    assert _episodes_alerte(np.zeros(5, dtype=int)) == []


def test_episodes_alerte_toutes_alertes():
    assert _episodes_alerte(np.ones(5, dtype=int)) == [(0, 4)]


def test_episodes_alerte_plusieurs_series():
    a = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert _episodes_alerte(a) == [(1, 2), (5, 5), (7, 9)]


def test_episodes_alerte_rearm_scinde_series_longues():
    a = np.ones(10, dtype=int)
    assert _episodes_alerte(a, pas_max_episode=4) == [(0, 3), (4, 7), (8, 9)]
    # pas_max_episode=None désactive explicitement le redécoupage
    assert _episodes_alerte(a, pas_max_episode=None) == [(0, 9)]


def test_episodes_alerte_pas_max_defaut_scinde_alerte_dun_mois():
    """Un modèle en alerte permanente pendant 300 jours ne doit pas compter comme 1 épisode."""
    n = 48 * 300  # 300 jours de pas 30 min, en alerte continue
    a = np.ones(n, dtype=int)
    episodes = _episodes_alerte(a)
    assert len(episodes) == 300  # un épisode par tranche de 24h (délai de réarmement par défaut)


def _creer_df_machine_unique(n: int, index_panne: int, mode_panne: str = "TWF") -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=n, freq="30min")
    panne = np.zeros(n, dtype=int)
    mode = np.full(n, "", dtype=object)
    panne[index_panne] = 1
    mode[index_panne] = mode_panne
    return pd.DataFrame({"id_machine": ["M1"] * n, "timestamp": timestamps, "panne": panne, "mode_panne": mode})


def test_metriques_evenementielles_attrape_panne_avec_episode_precedent():
    """Épisode qui précède directement la panne -> attrapée ; épisode isolé plus tôt -> fausse alerte."""
    n = 20
    df = _creer_df_machine_unique(n, index_panne=15)
    probas = np.zeros(n)
    probas[10:15] = 0.9  # épisode adjacent à la panne (idx 10-14)
    probas[0:3] = 0.9  # épisode isolé, hors fenêtre pas_horizon=10 -> [5, 15]

    metrics = metriques_evenementielles(df, probas, threshold=0.5, pas_horizon=10)

    assert metrics["nb_pannes"] == 1
    assert metrics["nb_attrapees"] == 1
    assert metrics["recall_evenementiel"] == 1.0
    assert metrics["nb_episodes"] == 2
    assert metrics["nb_episodes_fausses_alertes"] == 1
    assert metrics["preavis_heures"] == pytest.approx([2.5])


def test_metriques_evenementielles_manque_panne_sans_episode():
    n = 20
    df = _creer_df_machine_unique(n, index_panne=15)
    probas = np.zeros(n)

    metrics = metriques_evenementielles(df, probas, threshold=0.5, pas_horizon=10)

    assert metrics["nb_pannes"] == 1
    assert metrics["nb_attrapees"] == 0
    assert metrics["recall_evenementiel"] == 0.0
    assert metrics["nb_episodes"] == 0


def test_metriques_evenementielles_alerte_permanente_non_notee_parfaite():
    """
    Un modèle qui alerte 100 % du temps doit être lourdement pénalisé par le taux de
    fausses alertes, pas récompensé. Sans délai de réarmement, une alerte permanente
    ne forme qu'UN épisode sur toute la période -> FA/machine/jour proche de 0, un
    score en apparence excellent pour un modèle inutilisable en pratique.
    """
    n = 48 * 10  # 10 jours, alerte permanente, aucune panne
    df = pd.DataFrame(
        {
            "id_machine": ["M1"] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "panne": np.zeros(n, dtype=int),
            "mode_panne": np.full(n, "", dtype=object),
        }
    )
    probas = np.ones(n)

    metrics = metriques_evenementielles(df, probas, threshold=0.5, pas_horizon=10)

    assert metrics["nb_episodes"] == 10  # un épisode par tranche de 24h (délai de réarmement)
    assert metrics["taux_fausses_alertes_par_machine_jour"] == pytest.approx(1.0, rel=0.01)


def test_metriques_evenementielles_taux_fausses_alertes_par_machine_jour():
    n = 48 * 2  # 2 jours pile, pas de 30 min
    df = pd.DataFrame(
        {
            "id_machine": ["M1"] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "panne": np.zeros(n, dtype=int),
            "mode_panne": np.full(n, "", dtype=object),
        }
    )
    probas = np.zeros(n)
    probas[0:4] = 0.9  # 1 épisode de 2h, aucune panne -> 1 fausse alerte

    metrics = metriques_evenementielles(df, probas, threshold=0.5, pas_horizon=10)

    assert metrics["nb_episodes_fausses_alertes"] == 1
    assert metrics["taux_fausses_alertes_par_machine_jour"] == pytest.approx(0.5, rel=0.05)


def test_trouver_seuil_pour_budget_fa_repli_si_aucun_seuil_respecte_budget():
    """Budget de 0 fausse alerte impossible à tenir dès qu'il y a du bruit -> repli, pas de crash."""
    n = 20
    df = _creer_df_machine_unique(n, index_panne=15)
    probas = np.zeros(n)
    probas[10:15] = 0.9  # attrape la panne
    probas[0:2] = 0.9  # fausse alerte isolée, inévitable avec un seul seuil disponible

    meilleur_seuil, meilleures_metriques = trouver_seuil_pour_budget_fa(
        df, probas, thresholds=np.array([0.5]), pas_horizon=10, budget_par_machine_jour=0.0
    )

    assert meilleur_seuil == 0.5
    assert meilleures_metriques["taux_fausses_alertes_par_machine_jour"] > 0.0


def test_trouver_seuil_pour_budget_fa_maximise_recall_dans_budget():
    """Parmi les seuils respectant le budget, celui qui maximise le recall événementiel est choisi."""
    n = 20
    df = _creer_df_machine_unique(n, index_panne=15)
    probas = np.zeros(n)
    probas[10:15] = 0.6  # épisode adjacent à la panne, alerte à seuil 0.5 mais pas 0.7

    meilleur_seuil, meilleures_metriques = trouver_seuil_pour_budget_fa(
        df, probas, thresholds=np.array([0.5, 0.7]), pas_horizon=10, budget_par_machine_jour=10.0
    )

    assert meilleur_seuil == 0.5
    assert meilleures_metriques["recall_evenementiel"] == 1.0
