from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.features import calculer_features
from src.score import niveaux, scorer
from tests.conftest import serie_machine


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
