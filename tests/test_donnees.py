import numpy as np
import pandas as pd

from src.donnees import DATE_TEST, DATE_VALIDATION, HORIZON, MACHINES_RESERVEES, calculer_cible, decouper, signaler_anomalies
from src.features import calculer_features
from tests.conftest import serie_machine


def test_cible_marque_lhorizon_avant_une_panne_previsible():
    serie = serie_machine("M", n=400)
    serie.loc[300, ["panne", "mode_panne"]] = [1, "TWF"]
    cible = calculer_cible(calculer_features(serie))
    pas_horizon = int(HORIZON / pd.Timedelta("30min"))
    assert cible[300 - pas_horizon + 1 : 301].all()
    assert cible[: 300 - pas_horizon + 1].sum() == 0
    assert cible[301:].sum() == 0


def test_cible_ignore_les_pannes_sans_precurseur():
    serie = serie_machine("M", n=100)
    serie.loc[50, ["panne", "mode_panne"]] = [1, "PWF"]
    assert calculer_cible(calculer_features(serie)).sum() == 0


def test_decoupage_tampon_et_machines_reservees():
    horodatages = pd.date_range("2024-01-01", "2025-06-30", freq="6h")
    df = pd.DataFrame(
        {
            "timestamp": np.tile(horodatages, 2),
            "id_machine": np.repeat(["GDIZ_TEX_M_01", MACHINES_RESERVEES[0]], len(horodatages)),
        }
    )
    train, validation, test = decouper(df)

    assert train["timestamp"].max() < DATE_VALIDATION - HORIZON
    assert validation["timestamp"].min() >= DATE_VALIDATION
    assert validation["timestamp"].max() < DATE_TEST - HORIZON
    assert test["timestamp"].min() >= DATE_TEST
    assert not pd.concat([train, validation])["id_machine"].isin(MACHINES_RESERVEES).any()
    assert test["id_machine"].isin(MACHINES_RESERVEES).any()


def test_signaler_anomalies_compte_sans_bloquer():
    df = pd.DataFrame({"vibration": [1.0, 99999.0, np.nan], "type_machine": ["L", "X", "H"]})
    assert signaler_anomalies(df) == {"vibration": 1, "type_machine": 1}
