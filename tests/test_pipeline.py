import numpy as np
import pandas as pd
import pytest


def test_chargement_modele(predicteur):
    """Vérifie que le modèle est bien chargé en mémoire."""
    assert predicteur.pipeline is not None
    assert hasattr(predicteur.pipeline, "predict")


def test_format_sortie(predicteur, machine_normale):
    """Vérifie que la sortie respecte le contrat de données."""
    res = predicteur.predire(machine_normale)

    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"prediction", "probabilite_panne"}
    assert res["prediction"].dtype in [np.int64, np.int32, int]
    assert 0 <= res["probabilite_panne"].iloc[0] <= 1


def test_robustesse_colonnes_absentes(predicteur):
    """Vérifie que le pipeline lève une erreur si une colonne cruciale manque."""
    data_incomplète = pd.DataFrame({"temp": [25]})
    with pytest.raises(Exception):
        predicteur.predire(data_incomplète)


def test_sensibilite_surchauffe(predicteur, machine_en_surchauffe):
    """Non régression : LightGBM doit détecter un cas de panne évident.
    """
    res = predicteur.predire(machine_en_surchauffe)
    assert res["probabilite_panne"].iloc[0] > predicteur.threshold * 2
    assert res["prediction"].iloc[0] == 1


def test_stabilite_normal(predicteur, machine_normale):
    """Vérifie qu'une machine saine est bien prédite comme saine."""
    res = predicteur.predire(machine_normale)
    assert res["prediction"].iloc[0] == 0
    assert res["probabilite_panne"].iloc[0] < 0.3
