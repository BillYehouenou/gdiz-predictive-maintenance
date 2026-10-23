import pandas as pd
import pytest

from src.explain import FAMILLES_FEATURES, ExplicateurCause, famille_de
from src.features import FEATURES


def test_format_sortie(predicteur, machine_normale):
    res = predicteur.predire(machine_normale)
    assert list(res.columns) == ["prediction", "probabilite_panne"]
    assert 0 <= res["probabilite_panne"].iloc[0] <= 1


def test_colonnes_manquantes_levent_une_erreur(predicteur):
    with pytest.raises(KeyError):
        predicteur.predire(pd.DataFrame({"temp": [25]}))


def test_machine_saine_sous_le_seuil(predicteur, machine_normale):
    assert predicteur.predire(machine_normale)["prediction"].iloc[0] == 0


def test_machine_degradee_en_alerte(predicteur, machine_degradee):
    res = predicteur.predire(machine_degradee)
    assert res["probabilite_panne"].iloc[0] > predicteur.threshold * 2


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
