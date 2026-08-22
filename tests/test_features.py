import pandas as pd

from src.features import compute_failure_cause_label


def test_compute_failure_cause_label_assigns_next_failure_mode_within_horizon():
    """
    Étage 2 (diagnostic de cause) : chaque ligne doit porter la cause de la PROCHAINE
    panne prédictible si elle est dans l'horizon, None sinon — vérifié à la main sur
    un exemple à 2 pannes de causes différentes sur la même machine.
    """
    n = 20
    df = pd.DataFrame(
        {
            "machine_id": ["M1"] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "target": [0] * 10 + [1] + [0] * 5 + [1] + [0] * 3,
            "failure_mode": [""] * 10 + ["HDF"] + [""] * 5 + ["OSF"] + [""] * 3,
        }
    )

    label = compute_failure_cause_label(df, horizon_hours=3, failure_modes=("TWF", "HDF", "OSF"))  # horizon_steps=6

    # Fenêtre HDF : 6 pas avant l'index 10 (inclus) -> indices 4-10
    assert label.iloc[3] is None
    assert (label.iloc[4:11] == "HDF").all()
    # Fenêtre OSF : 6 pas avant l'index 16 (inclus) -> indices 11-16 (11 n'est pas dans la fenêtre HDF)
    assert (label.iloc[11:17] == "OSF").all()
    # Après la dernière panne : pas de panne future -> None
    assert label.iloc[17:].isna().all()


def test_compute_failure_cause_label_ignores_non_predictable_modes():
    n = 15
    df = pd.DataFrame(
        {
            "machine_id": ["M1"] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "target": [0] * 10 + [1] + [0] * 4,
            "failure_mode": [""] * 10 + ["PWF"] + [""] * 4,
        }
    )

    label = compute_failure_cause_label(df, horizon_hours=3, failure_modes=("TWF", "HDF", "OSF"))

    assert label.isna().all()
