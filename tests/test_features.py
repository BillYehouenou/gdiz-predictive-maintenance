import numpy as np
import pandas as pd

from src.features import FEATURES, LIGNES_HISTORIQUE, calculer_features, preparer_X
from tests.conftest import serie_machine


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
