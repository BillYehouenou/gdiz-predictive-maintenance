import re
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from ui.priorites import COLONNES_PRIORITES, calculer_priorites_parc


def _historique_machine(id_machine: str, type_machine: str, usure_finale: float = 50.0, n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_machine": [id_machine] * n,
            "type_machine": [type_machine] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "usure_outil": np.linspace(10, usure_finale, n),
            "vibration": np.full(n, 1.5),
            "temperature_process": np.full(n, 45.0),
            "temperature_ambiante": np.full(n, 27.0),
            "concentration_poussiere": np.full(n, 40.0),
            "couple": np.full(n, 30.0),
            "encrassement": np.full(n, 20.0),
            "saison_benin": np.zeros(n, dtype=int),
            "vitesse_rotation": np.full(n, 1500.0),
            "niveau_activite": np.full(n, 0.5),
            "humidite": np.full(n, 65.0),
            "indicateur_pluie": np.zeros(n, dtype=int),
            "indicateur_coupure": np.zeros(n, dtype=int),
            "niveau_tension": np.full(n, 220.0),
            "stabilite_tension": np.full(n, 90.0),
        }
    )


def _predicteur_factice(seuil: float, proba_par_machine: dict) -> MagicMock:
    predicteur = MagicMock()
    predicteur.threshold = seuil

    def predire(df):
        # Le test contrôle la probabilité renvoyée en la faisant dépendre de l'usure
        # de la dernière ligne, pour que chaque machine du test ait une proba distincte.
        usure = float(df["usure_outil"].iloc[0])
        proba = proba_par_machine.get(round(usure, 4), 0.01)
        return pd.DataFrame({"prediction": [int(proba >= seuil)], "probabilite_panne": [proba]})

    predicteur.predire.side_effect = predire
    return predicteur


def _explicateur_factice(cause: str = "Usure de l'outil") -> MagicMock:
    explicateur = MagicMock()
    explicateur.cause_probable.return_value = [{"cause_principale": cause, "score": 1.0, "detail": {}}]
    return explicateur


def test_calculer_priorites_parc_ne_garde_que_les_machines_en_alerte():
    m1 = _historique_machine("M1", "L", usure_finale=50.0)
    m2 = _historique_machine("M2", "M", usure_finale=80.0)
    df_historique = pd.concat([m1, m2], ignore_index=True)

    proba_par_machine = {round(m1["usure_outil"].iloc[-1], 4): 0.9, round(m2["usure_outil"].iloc[-1], 4): 0.01}
    predicteur = _predicteur_factice(seuil=0.1, proba_par_machine=proba_par_machine)
    explicateur = _explicateur_factice()

    resultat = calculer_priorites_parc(df_historique, predicteur, explicateur)

    assert list(resultat["id_machine"]) == ["M1"]
    assert list(resultat.columns) == COLONNES_PRIORITES


def test_calculer_priorites_parc_trie_par_risque_decroissant():
    m1 = _historique_machine("M1", "L", usure_finale=50.0)
    m2 = _historique_machine("M2", "M", usure_finale=80.0)
    df_historique = pd.concat([m1, m2], ignore_index=True)

    proba_par_machine = {round(m1["usure_outil"].iloc[-1], 4): 0.5, round(m2["usure_outil"].iloc[-1], 4): 0.9}
    predicteur = _predicteur_factice(seuil=0.1, proba_par_machine=proba_par_machine)
    explicateur = _explicateur_factice()

    resultat = calculer_priorites_parc(df_historique, predicteur, explicateur)

    assert list(resultat["id_machine"]) == ["M2", "M1"]


def test_calculer_priorites_parc_fenetre_qualitative_pas_dheure_precise():
    """Point 34 roadmap : jamais d'heure/délai précis inventé — seulement une
    fenêtre qualitative, différenciée par intensité du risque."""
    m1 = _historique_machine("M1", "L")
    proba = round(m1["usure_outil"].iloc[-1], 4)
    predicteur = _predicteur_factice(seuil=0.1, proba_par_machine={proba: 0.9})
    explicateur = _explicateur_factice()

    resultat = calculer_priorites_parc(m1, predicteur, explicateur)

    fenetre = resultat.iloc[0]["fenetre"]
    assert "Risque élevé" in fenetre
    # Aucune heure précise (ex. "37h", "14:32") — seulement une fenêtre qualitative en jours.
    assert not re.search(r"\d+\s*h\b|\d+:\d+", fenetre)


def test_calculer_priorites_parc_aucune_alerte_retourne_dataframe_vide():
    m1 = _historique_machine("M1", "L")
    predicteur = _predicteur_factice(seuil=0.99, proba_par_machine={})
    explicateur = _explicateur_factice()

    resultat = calculer_priorites_parc(m1, predicteur, explicateur)

    assert resultat.empty
    assert list(resultat.columns) == COLONNES_PRIORITES
