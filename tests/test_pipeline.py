import numpy as np
import pandas as pd
import pytest


def test_chargement_modele(predictor):
    """Vérifie que le modèle est bien chargé en mémoire."""
    assert predictor.pipeline is not None
    assert hasattr(predictor.pipeline, "predict")


def test_format_sortie(predictor, machine_normale):
    """Vérifie que la sortie respecte le contrat de données."""
    res = predictor.predict(machine_normale)

    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"prediction", "failure_probability"}
    assert res["prediction"].dtype in [np.int64, np.int32, int]
    assert 0 <= res["failure_probability"].iloc[0] <= 1


def test_robustesse_colonnes_absentes(predictor):
    """Vérifie que le pipeline lève une erreur si une colonne cruciale manque."""
    data_incomplète = pd.DataFrame({"temp": [25]})
    with pytest.raises(Exception):
        predictor.predict(data_incomplète)


def test_sensibilite_surchauffe(predictor, machine_en_surchauffe):
    """Non régression : LightGBM avec seuil F2 doit détecter un cas de panne évident."""
    res = predictor.predict(machine_en_surchauffe)
    assert res["failure_probability"].iloc[0] > 0.5
    assert res["prediction"].iloc[0] == 1


def test_stabilite_normal(predictor, machine_normale):
    """Vérifie qu'une machine saine est bien prédite comme saine."""
    res = predictor.predict(machine_normale)
    assert res["prediction"].iloc[0] == 0
    assert res["failure_probability"].iloc[0] < 0.3
