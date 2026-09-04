import logging

import pandas as pd

from src.validation import valider_donnees_brutes


def test_valider_donnees_brutes_laisse_passer_donnees_saines():
    df = pd.DataFrame({"usure_outil": [10.0, 50.0, 90.0], "type_machine": ["L", "M", "H"]})
    resultat = valider_donnees_brutes(df)
    assert len(resultat) == 3


def test_valider_donnees_brutes_tolere_derive_de_calibration():
    """
    La dérive de calibration injectée par le générateur n'est pas bornée par construction.
    usure_outil peut légitimement passer sous 0 dans le dataset officiel. La validation ne doit pas planter dessus.
    """
    df = pd.DataFrame({"usure_outil": [-11.4, 50.0], "vibration": [-0.8, 2.0]})
    resultat = valider_donnees_brutes(df)
    assert len(resultat) == 2


def test_valider_donnees_brutes_detecte_categorie_invalide(caplog):
    df = pd.DataFrame({"type_machine": ["H", "INEXISTANT"]})
    with caplog.at_level(logging.WARNING):
        valider_donnees_brutes(df)
    assert any("type_machine" in message for message in caplog.messages)


def test_valider_donnees_brutes_detecte_valeur_physiquement_impossible(caplog):
    df = pd.DataFrame({"usure_outil": [50.0, 99999.0]})
    with caplog.at_level(logging.WARNING):
        valider_donnees_brutes(df)
    assert any("usure_outil" in message for message in caplog.messages)


def test_valider_donnees_brutes_ignore_colonnes_absentes():
    """Le schéma couvre plus de colonnes que ce qu'un appelant peut fournir (ex.
    historique partiel API) — ne doit jamais exiger une colonne manquante."""
    df = pd.DataFrame({"couple": [20.0]})
    resultat = valider_donnees_brutes(df)
    assert len(resultat) == 1
