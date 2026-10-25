import numpy as np
import pandas as pd

from src.drift import VARIABLES, performance_recente, psi, verifier_derive


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


def test_performance_recente_ne_compte_que_la_periode_recente():
    """Une panne ancienne est ignorée ; la panne récente, précédée d'une alerte, est annoncée."""
    horodatages = pd.date_range("2025-01-01", periods=48 * 90, freq="30min")
    df = pd.DataFrame(
        {"id_machine": "M1", "type_machine": "M", "timestamp": horodatages, "etat": "marche", "panne": 0, "mode_panne": ""}
    )
    df["probabilite_panne"] = 0.01
    for i in (48 * 5, 48 * 80):  # une panne au 5e jour (hors des 60 derniers), une au 80e
        df.loc[i, ["panne", "mode_panne", "etat"]] = [1, "TWF", "panne"]
    df.loc[48 * 80 - 20 : 48 * 80 - 1, "probabilite_panne"] = 0.5
    p = performance_recente(df, seuil=0.1, jours=60)
    assert (p["nb_pannes"], p["nb_attrapees"]) == (1, 1)
    assert p["nb_episodes_fausses_alertes"] == 0
