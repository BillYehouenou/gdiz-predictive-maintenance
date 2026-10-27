import numpy as np
import pandas as pd

from src.donnees import (
    DATE_TEST,
    DATE_VALIDATION,
    FEATURES,
    HORIZON,
    LIGNES_HISTORIQUE,
    MACHINES_RESERVEES,
    calculer_cible,
    calculer_features,
    decouper,
    preparer_X,
    signaler_anomalies,
)
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


def _serie_bruitee(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return serie_machine(
        "M", n=n, vibration=rng.normal(1.5, 0.3, n), temperature_process=rng.normal(58, 2, n), couple=rng.normal(30, 3, n)
    )


def test_inference_sur_24h_identique_a_lentrainement():
    """L'API ne dispose que des dernières 24h : la dernière ligne doit être identique au calcul sur la série complète."""
    serie = _serie_bruitee()
    complet = calculer_features(serie).iloc[-1][FEATURES]
    fenetre = calculer_features(serie.tail(LIGNES_HISTORIQUE + 1)).iloc[-1][FEATURES]
    pd.testing.assert_series_equal(complet, fenetre, check_names=False)


def test_aucune_fuite_du_futur():
    serie = _serie_bruitee()
    passe = calculer_features(serie.head(100))[FEATURES]
    avec_futur = calculer_features(serie).head(100)[FEATURES]
    pd.testing.assert_frame_equal(passe, avec_futur)


def test_machines_calculees_independamment():
    a, b = serie_machine("L", vibration=1.0), serie_machine("H", vibration=5.0)
    resultat = calculer_features(pd.concat([a, b]))
    assert resultat.loc[resultat["id_machine"] == "M_L", "vibration_max_24h"].max() == 1.0


def test_preparer_x_categorie_stable_meme_avec_un_seul_type():
    X = preparer_X(calculer_features(serie_machine("H")))
    assert list(X.columns) == FEATURES
    assert list(X["type_machine"].cat.categories) == ["L", "M", "H"]


def test_moyennes_ignorent_les_releves_a_larret():
    """Capteurs à zéro le week-end : la moyenne du lundi ne doit voir que les relevés en marche."""
    serie = serie_machine("M", n=48, vibration=2.0).assign(etat="marche")
    serie.loc[:23, ["vibration", "etat"]] = [0.0, "arret"]
    assert calculer_features(serie)["vibration_moyenne_24h"].iloc[-1] == 2.0
