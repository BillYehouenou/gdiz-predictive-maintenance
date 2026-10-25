from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.config import NOM_MODELE
from src.features import calculer_features, preparer_X
from src.train import LGBMLogOdds, nouveau_modele, promouvoir_si_meilleur
from tests.conftest import serie_machine


def _client(score_champion: float | None) -> MagicMock:
    client = MagicMock()
    if score_champion is None:
        client.get_model_version_by_alias.side_effect = Exception("RESOURCE_DOES_NOT_EXIST")
    else:
        client.get_run.return_value.data.metrics = {"recall_evenementiel": score_champion}
    return client


def test_promotion_automatique_sans_champion():
    client = _client(None)
    assert promouvoir_si_meilleur(client, "1", 0.10)
    client.set_registered_model_alias.assert_called_once_with(NOM_MODELE, "champion", "1")


def test_champion_conserve_si_score_inferieur():
    client = _client(0.50)
    assert not promouvoir_si_meilleur(client, "4", 0.20)
    client.set_registered_model_alias.assert_not_called()


def test_promotion_si_meilleur():
    client = _client(0.80)
    assert promouvoir_si_meilleur(client, "4", 0.95)
    client.set_registered_model_alias.assert_called_once_with(NOM_MODELE, "champion", "4")


@pytest.fixture(scope="module")
def jeu_entrainement() -> tuple[pd.DataFrame, np.ndarray]:
    """Deux machines, valeurs manquantes comprises ; la cible suit les heures d'outil."""
    rng = np.random.default_rng(0)
    series = [
        serie_machine(t, n=300, heures_depuis_changement_outil=np.linspace(0, 6000, 300), vibration=rng.normal(1.5, 0.3, 300))
        for t in ("L", "H")
    ]
    df = calculer_features(pd.concat(series))
    df.loc[::17, "vibration"] = np.nan
    return preparer_X(df), (df["heures_depuis_changement_outil"] > 4500).astype(int).to_numpy()


def test_le_modele_retenu_sentraine_et_predit(jeu_entrainement):
    X, y = jeu_entrainement
    probas = nouveau_modele().fit(X, y).predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert not np.isnan(probas).any()


def test_lightgbm_expose_ses_log_odds(jeu_entrainement):
    X, y = jeu_entrainement
    modele = LGBMLogOdds(n_estimators=20, verbose=-1).fit(X, y)
    probas = modele.predict_proba(X)[:, 1]
    assert np.allclose(modele.decision_function(X), np.log(probas / (1 - probas)))


def test_calibration_sur_log_odds_sans_plancher(jeu_entrainement):
    """Les lignes que le modèle juge les plus sûres restent proches de 0 après calibration."""
    X, y = jeu_entrainement
    modele = LGBMLogOdds(n_estimators=20, verbose=-1).fit(X, y)
    calibre = CalibratedClassifierCV(FrozenEstimator(modele), method="sigmoid").fit(X, y)
    assert calibre.predict_proba(X)[:, 1].min() < 0.01
