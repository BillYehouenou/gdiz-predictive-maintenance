"""
Tests de la métrique événementielle : logique isolée du reste du pipeline
d'entraînement, donc vérifiable par calcul manuel plutôt que par supposition.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import _episodes_alerte, bilan_economique, metriques_evenementielles, trouver_seuil_pour_budget_fa
from src.metier import COUT_INSPECTION, gain_panne_evitee


@pytest.mark.parametrize(
    "alertes,pas_max,attendu",
    [
        ([0, 0, 0, 0, 0], 48, []),
        ([1, 1, 1, 1, 1], 48, [(0, 4)]),
        ([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], 48, [(1, 2), (5, 5), (7, 9)]),
        ([1] * 10, 4, [(0, 3), (4, 7), (8, 9)]),  # les séries longues sont redécoupées
        ([1] * 10, None, [(0, 9)]),  # None désactive le redécoupage
    ],
)
def test_episodes_alerte(alertes, pas_max, attendu):
    assert _episodes_alerte(np.array(alertes), pas_max_episode=pas_max) == attendu


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


def test_metriques_evenementielles_alerte_longue_dans_lhorizon_entierement_utile():
    """Alerte continue sur tout l'horizon, redécoupée : aucune tranche n'est fausse et le préavis part de la première."""
    n = 30
    df = _creer_df_machine_unique(n, index_panne=25)
    probas = np.zeros(n)
    probas[15:25] = 0.9  # deux tranches de 5 pas, toutes deux dans [15, 25]

    metrics = metriques_evenementielles(df, probas, threshold=0.5, pas_horizon=10, pas_max_episode=5)

    assert metrics["nb_episodes"] == 2
    assert metrics["nb_episodes_fausses_alertes"] == 0
    assert metrics["preavis_heures"] == pytest.approx([5.0])
    assert metrics["precision_alertes"] == 1.0


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


def test_aucune_alerte_sur_une_machine_qui_ne_tourne_pas():
    """Week-end ou réparation : même avec un risque élevé, pas d'épisode d'alerte."""
    n = 20
    df = _creer_df_machine_unique(n, index_panne=15).assign(etat="marche")
    df.loc[0:5, "etat"] = "arret"
    probas = np.zeros(n)
    probas[0:4] = 0.9  # risque élevé pendant l'arrêt : ignoré
    metrics = metriques_evenementielles(df, probas, threshold=0.5, pas_horizon=10)
    assert metrics["nb_episodes"] == 0


def test_precision_et_recall_avec_preavis_utile():
    """Deux pannes attrapées, l'une 1 h avant (trop tard), l'autre 5 h avant ; une fausse alerte."""
    n = 60
    df = _creer_df_machine_unique(n, index_panne=20)
    df.loc[50, ["panne", "mode_panne"]] = [1, "TWF"]
    probas = np.zeros(n)
    probas[18:20] = 0.9  # 1 h avant la première panne
    probas[40:42] = 0.9  # 5 h avant la seconde
    probas[2:4] = 0.9  # hors de tout horizon : fausse alerte
    metrics = metriques_evenementielles(df, probas, threshold=0.5, pas_horizon=12)
    assert metrics["recall_evenementiel"] == 1.0
    assert metrics["recall_preavis_4h"] == 0.5
    assert metrics["precision_alertes"] == pytest.approx(2 / 3)


def test_bilan_economique_valeur_des_pannes_moins_les_inspections():
    n = 30
    df = _creer_df_machine_unique(n, index_panne=15).assign(etat="marche", type_machine="M")
    df.loc[15:23, "etat"] = "panne"  # la ligne de panne puis 8 pas de réparation : 4 h
    probas = np.zeros(n)
    probas[12:15] = 0.9  # attrape la panne
    probas[0:2] = 0.9  # fausse alerte
    metrics = metriques_evenementielles(df, probas, threshold=0.5, pas_horizon=10)
    bilan = bilan_economique(df, metrics)
    assert bilan["valeur_pannes_evitees"] == pytest.approx(gain_panne_evitee("M", 4.0))
    assert bilan["gain_net"] == pytest.approx(gain_panne_evitee("M", 4.0) - COUT_INSPECTION)
