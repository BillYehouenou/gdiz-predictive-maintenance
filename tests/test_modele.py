from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.config import NOM_MODELE
from src.donnees import FEATURES, calculer_features, preparer_X
from src.modele import FAMILLES_FEATURES, ExplicateurCause, famille_de, niveaux, scorer
from src.train import METRIQUE_PROMOTION, LGBMLogOdds, nouveau_modele, promouvoir_si_meilleur
from tests.conftest import serie_machine


def test_machine_saine_sous_le_seuil(predicteur, machine_normale):
    res = predicteur.predire(machine_normale)
    assert list(res.columns) == ["prediction", "probabilite_panne"]
    assert res["prediction"].iloc[0] == 0


def test_machine_degradee_en_alerte(predicteur, machine_degradee):
    res = predicteur.predire(machine_degradee)
    assert res["prediction"].iloc[0] == 1


def test_cause_probable(predicteur, machine_normale, machine_degradee):
    resultats = ExplicateurCause(predicteur.modele).cause_probable(pd.concat([machine_normale, machine_degradee]))
    assert len(resultats) == 2
    for r in resultats:
        assert r["cause_principale"] in set(FAMILLES_FEATURES) | {"Aucun signal dominant"}
        assert all(score >= 0 for score in r["detail"].values())


def test_familles_couvrent_des_features_existantes():
    for famille, mesures in FAMILLES_FEATURES.items():
        assert all(m in FEATURES for m in mesures), famille
    assert famille_de("vibration_max_24h") == "Surcharge mécanique"
    assert famille_de("type_machine") is None


def test_niveaux_relatifs_au_seuil():
    assert list(niveaux(np.array([0.01, 0.05, 0.2]), seuil=0.04)) == ["normal", "surveillance", "eleve"]


def test_cause_calculee_seulement_sur_le_dernier_releve_des_machines_en_alerte():
    df = calculer_features(pd.concat([serie_machine("L", n=10), serie_machine("H", n=10)]))
    predicteur = MagicMock(threshold=0.1)
    # Machine L en alerte (0.5), machine H saine (0.01)
    predicteur.predire.return_value = pd.DataFrame({"probabilite_panne": np.where(df["id_machine"] == "M_L", 0.5, 0.01)})
    explicateur = MagicMock()
    explicateur.cause_probable.return_value = [{"cause_principale": "Usure et entretien"}]

    predictions = scorer(df, predicteur, explicateur)

    assert len(predictions) == len(df)
    assert predictions["cause_principale"].notna().sum() == 1
    derniere_l = df.index[df["id_machine"] == "M_L"][-1]
    assert predictions.loc[derniere_l, "cause_principale"] == "Usure et entretien"
    assert explicateur.cause_probable.call_args[0][0]["id_machine"].tolist() == ["M_L"]


def _client(score_champion: float | None) -> MagicMock:
    client = MagicMock()
    if score_champion is None:
        client.get_model_version_by_alias.side_effect = Exception("RESOURCE_DOES_NOT_EXIST")
    else:
        client.get_run.return_value.data.metrics = {METRIQUE_PROMOTION: score_champion}
    return client


def test_promotion_automatique_sans_champion():
    client = _client(None)
    assert promouvoir_si_meilleur(client, "1", 0.10)
    client.set_registered_model_alias.assert_called_once_with(NOM_MODELE, "champion", "1")


def test_champion_conserve_si_score_inferieur():
    client = _client(0.50)
    assert not promouvoir_si_meilleur(client, "4", 0.20)
    client.set_registered_model_alias.assert_not_called()


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


def test_calibration_sur_log_odds_sans_plancher(jeu_entrainement):
    """Les lignes que le modèle juge les plus sûres restent proches de 0 après calibration."""
    X, y = jeu_entrainement
    modele = LGBMLogOdds(n_estimators=20, verbose=-1).fit(X, y)
    calibre = CalibratedClassifierCV(FrozenEstimator(modele), method="sigmoid").fit(X, y)
    assert calibre.predict_proba(X)[:, 1].min() < 0.01
