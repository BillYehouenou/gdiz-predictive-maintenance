from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.check_drift import _psi, verifier_derive, verifier_derive_predictions


def test_psi_distributions_identiques_proche_de_zero():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 5000))
    production = pd.Series(rng.normal(0, 1, 2000))
    assert _psi(reference, production) < 0.1


def test_psi_distributions_decalees_eleve():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 5000))
    production = pd.Series(rng.normal(3, 1, 2000))
    assert _psi(reference, production) > 0.25


def test_psi_reference_vide_ne_plante_pas():
    assert _psi(pd.Series(dtype=float), pd.Series([1.0, 2.0])) == 0.0


def test_verifier_derive_sans_production_retourne_liste_vide():
    with patch("scripts.check_drift._charger_distribution_production", return_value=pd.DataFrame()):
        assert verifier_derive() == []


def test_verifier_derive_predictions_sans_production_retourne_none():
    with patch("scripts.check_drift._charger_distribution_production", return_value=pd.DataFrame()):
        assert verifier_derive_predictions() is None
