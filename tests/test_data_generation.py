import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.generate_synthetic_data import AgregateurDonnees, ConfigDonneesSynthetiques

COLONNES_CAPTEUR = [
    "usure_outil",
    "temperature_process",
    "vitesse_rotation",
    "couple",
    "vibration",
    "temperature_ambiante",
    "humidite",
    "concentration_poussiere",
]


@pytest.fixture
def df_capteur_factice() -> pd.DataFrame:
    n = 3000
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "id_machine": ["M1"] * n,
            "usure_outil": rng.uniform(0, 100, n),
            "temperature_process": rng.uniform(20, 90, n),
            "vitesse_rotation": rng.uniform(0, 2000, n),
            "couple": rng.uniform(0, 80, n),
            "vibration": rng.uniform(0, 10, n),
            "temperature_ambiante": rng.uniform(20, 40, n),
            "humidite": rng.uniform(40, 100, n),
            "concentration_poussiere": rng.uniform(0, 500, n),
        }
    )


def test_injecter_donnees_manquantes_compat_pandas3_copy_on_write(df_capteur_factice):
    original_cow = pd.options.mode.copy_on_write
    np.random.seed(42)
    try:
        pd.options.mode.copy_on_write = True
        agregateur = AgregateurDonnees(ConfigDonneesSynthetiques())
        result = agregateur.injecter_donnees_manquantes(df_capteur_factice)
    finally:
        pd.options.mode.copy_on_write = original_cow

    # Sur 3000 pas, au moins une rafale de perte de paquets est quasi certaine
    assert result[COLONNES_CAPTEUR].isna().sum().sum() > 0
    assert df_capteur_factice[COLONNES_CAPTEUR].isna().sum().sum() == 0


def test_injecter_donnees_manquantes_produit_rafales_correlees(df_capteur_factice):
    """Les valeurs manquantes doivent former des rafales corrélées dans le temps."""
    np.random.seed(7)
    agregateur = AgregateurDonnees(ConfigDonneesSynthetiques())
    result = agregateur.injecter_donnees_manquantes(df_capteur_factice)

    is_nan = result["vibration"].isna().to_numpy().astype(int)
    assert is_nan.sum() > 0, "aucune rafale générée sur cet essai ; augmenter la taille de l'échantillon"

    diff = np.diff(np.concatenate([[0], is_nan, [0]]))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    run_lengths = ends - starts

    # La majorité des rafales doivent durer plus d'un pas 
    assert (run_lengths > 1).mean() > 0.5
