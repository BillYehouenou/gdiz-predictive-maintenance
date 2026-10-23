import numpy as np
import pandas as pd

from src.drift import VARIABLES, psi, verifier_derive


def test_psi_distributions_identiques_proche_de_zero():
    rng = np.random.default_rng(42)
    assert psi(pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(0, 1, 2000))) < 0.1


def test_psi_distributions_decalees_eleve():
    rng = np.random.default_rng(42)
    assert psi(pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(3, 1, 2000))) > 0.25


def test_psi_reference_vide_ne_plante_pas():
    assert psi(pd.Series(dtype=float), pd.Series([1.0, 2.0])) == 0.0


def test_verifier_derive_detecte_un_decalage_recent():
    rng = np.random.default_rng(0)
    horodatages = pd.date_range("2024-01-01", "2025-12-31", freq="6h")
    df = pd.DataFrame({v: rng.normal(0, 1, len(horodatages)) for v in VARIABLES} | {"timestamp": horodatages})
    df.loc[df["timestamp"] > "2025-12-01", "vibration"] += 3
    rapport = verifier_derive(df).set_index("variable")
    assert rapport.loc["vibration", "drift"]
    assert not rapport.drop(index="vibration")["drift"].any()
