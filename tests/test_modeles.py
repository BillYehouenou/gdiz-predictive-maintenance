import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.features import calculer_features, preparer_X
from src.modeles import CANDIDATS, MODELE_RETENU, LGBMLogOdds, RegleSeuil
from tests.conftest import serie_machine


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


@pytest.mark.parametrize("nom", list(CANDIDATS))
def test_chaque_candidat_sentraine_et_predit(nom, jeu_entrainement):
    X, y = jeu_entrainement
    probas = CANDIDATS[nom]().fit(X, y).predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert not np.isnan(probas).any()


def test_regle_score_croissant_avec_les_heures(jeu_entrainement):
    X, y = jeu_entrainement
    scores = RegleSeuil().fit(X, y).predict_proba(X)[:, 1]
    heures = X["heures_depuis_changement_outil"].to_numpy()
    assert np.array_equal(np.argsort(scores, kind="stable"), np.argsort(heures, kind="stable"))


def test_modele_retenu_fait_partie_des_candidats():
    assert MODELE_RETENU in CANDIDATS


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
