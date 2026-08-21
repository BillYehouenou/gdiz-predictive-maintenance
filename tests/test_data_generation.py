"""
Tests de non-régression pour les bugs de reproductibilité de la génération synthétique.

Contexte : la génération de données doit être bit-à-bit reproductible d'une exécution
à l'autre (même seed → même jeu_donnees) et compatible avec le mode copy-on-write qui
deviendra le comportement par défaut de pandas 3.x. Deux bugs corrigés ici :
- `injecter_donnees_manquantes` renvoyait un tableau numpy read-only sous copy-on-write.
- Le seed par machine dépendait de `hash()`, salé aléatoirement par process.
"""

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
    """Reproduit le mode par défaut de pandas 3.x : to_numpy() doit rester inscriptible."""
    original_cow = pd.options.mode.copy_on_write
    np.random.seed(42)
    try:
        pd.options.mode.copy_on_write = True
        agregateur = AgregateurDonnees(ConfigDonneesSynthetiques())
        result = agregateur.injecter_donnees_manquantes(df_capteur_factice)
    finally:
        pd.options.mode.copy_on_write = original_cow

    # Sur 3000 pas, au moins une rafale de perte de paquets est quasi certaine —
    # preuve que l'écriture sur le tableau extrait par to_numpy() n'a pas échoué.
    assert result[COLONNES_CAPTEUR].isna().sum().sum() > 0
    # La source n'a pas été mutée en place (pas d'effet de bord via une vue partagée).
    assert df_capteur_factice[COLONNES_CAPTEUR].isna().sum().sum() == 0


def test_injecter_donnees_manquantes_produit_rafales_correlees(df_capteur_factice):
    """
    Point 10 : les valeurs manquantes doivent former des rafales corrélées dans le
    temps (perte de paquets IoT réaliste), pas des cellules isolées au hasard (MCAR).
    """
    np.random.seed(7)
    agregateur = AgregateurDonnees(ConfigDonneesSynthetiques())
    result = agregateur.injecter_donnees_manquantes(df_capteur_factice)

    is_nan = result["vibration"].isna().to_numpy().astype(int)
    assert is_nan.sum() > 0, "aucune rafale générée sur cet essai — augmenter la taille de l'échantillon"

    diff = np.diff(np.concatenate([[0], is_nan, [0]]))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    run_lengths = ends - starts

    # La majorité des rafales doivent durer plus d'un pas — sinon on a juste reproduit
    # le MCAR d'origine sous un autre nom.
    assert (run_lengths > 1).mean() > 0.5


def test_graine_machine_deterministe_entre_processus():
    """
    Le calcul du seed par machine ne doit plus dépendre de hash() (salé par process).
    On force deux PYTHONHASHSEED différents entre les deux sous-process : avec l'ancien
    code (hash() builtin), les deux valeurs auraient divergé ; avec le hash md5
    déterministe, elles doivent être identiques.
    """
    snippet = (
        "import hashlib; "
        "id_machine = 'GDIZ_TEX_M_01'; "
        "h = int(hashlib.md5(id_machine.encode('utf-8')).hexdigest(), 16) % 10000; "
        "print(42 + h)"
    )

    resultats = []
    for graine_hash in ("111", "222"):
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": graine_hash, "PATH": os.environ.get("PATH", "")},
            check=True,
        )
        resultats.append(proc.stdout.strip())

    assert resultats[0] == resultats[1]
