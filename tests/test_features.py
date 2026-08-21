import pandas as pd

from src.features import calculer_label_cause_panne


def test_calculer_label_cause_panne_assigne_prochain_mode_panne_dans_horizon():
    """
    Étage 2 (diagnostic de cause) : chaque ligne doit porter la cause de la PROCHAINE
    panne prédictible si elle est dans l'horizon, None sinon — vérifié à la main sur
    un exemple à 2 pannes de causes différentes sur la même machine.
    """
    n = 20
    df = pd.DataFrame(
        {
            "id_machine": ["M1"] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "panne": [0] * 10 + [1] + [0] * 5 + [1] + [0] * 3,
            "mode_panne": [""] * 10 + ["HDF"] + [""] * 5 + ["OSF"] + [""] * 3,
        }
    )

    label = calculer_label_cause_panne(df, heures_horizon=3, modes_panne=("TWF", "HDF", "OSF"))  # pas_horizon=6

    # Fenêtre HDF : 6 pas avant l'index 10 (inclus) -> indices 4-10
    assert label.iloc[3] is None
    assert (label.iloc[4:11] == "HDF").all()
    # Fenêtre OSF : 6 pas avant l'index 16 (inclus) -> indices 11-16 (11 n'est pas dans la fenêtre HDF)
    assert (label.iloc[11:17] == "OSF").all()
    # Après la dernière panne : pas de panne future -> None
    assert label.iloc[17:].isna().all()


def test_calculer_label_cause_panne_ignore_modes_non_predictibles():
    n = 15
    df = pd.DataFrame(
        {
            "id_machine": ["M1"] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "panne": [0] * 10 + [1] + [0] * 4,
            "mode_panne": [""] * 10 + ["PWF"] + [""] * 4,
        }
    )

    label = calculer_label_cause_panne(df, heures_horizon=3, modes_panne=("TWF", "HDF", "OSF"))

    assert label.isna().all()
