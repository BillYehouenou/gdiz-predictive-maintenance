import re
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from ui.priorites import COLONNES_PRIORITES, calculer_priorites_parc


def _historique_machine(id_machine: str, type_machine: str, heures_finales: float = 5000.0, n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_machine": [id_machine] * n,
            "type_machine": [type_machine] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "heures_depuis_changement_outil": np.linspace(100, heures_finales, n),
            "vibration": np.full(n, 1.5),
            "temperature_process": np.full(n, 45.0),
            "temperature_ambiante": np.full(n, 27.0),
            "concentration_poussiere": np.full(n, 40.0),
            "couple": np.full(n, 30.0),
            "heures_depuis_maintenance": np.full(n, 300.0),
            "vitesse_rotation": np.full(n, 1500.0),
            "niveau_activite": np.full(n, 0.5),
            "humidite": np.full(n, 65.0),
            "precipitation_mm": np.zeros(n),
            "indicateur_coupure": np.zeros(n, dtype=int),
            "niveau_tension": np.full(n, 220.0),
        }
    )


def _predicteur_factice(seuil: float, proba_par_machine: dict) -> MagicMock:
    predicteur = MagicMock()
    predicteur.threshold = seuil

    def predire(df):
        heures = float(df["heures_depuis_changement_outil"].iloc[0])
        proba = proba_par_machine.get(round(heures, 4), 0.01)
        return pd.DataFrame({"prediction": [int(proba >= seuil)], "probabilite_panne": [proba]})

    predicteur.predire.side_effect = predire
    return predicteur


def _explicateur_factice(cause: str = "Usure et entretien") -> MagicMock:
    explicateur = MagicMock()
    explicateur.cause_probable.return_value = [{"cause_principale": cause, "score": 1.0, "detail": {}}]
    return explicateur


def test_calculer_priorites_parc_ne_garde_que_les_machines_en_alerte():
    m1 = _historique_machine("M1", "L", heures_finales=5000.0)
    m2 = _historique_machine("M2", "M", heures_finales=8000.0)
    df_historique = pd.concat([m1, m2], ignore_index=True)

    proba_par_machine = {
        round(m1["heures_depuis_changement_outil"].iloc[-1], 4): 0.9,
        round(m2["heures_depuis_changement_outil"].iloc[-1], 4): 0.01,
    }
    predicteur = _predicteur_factice(seuil=0.1, proba_par_machine=proba_par_machine)
    explicateur = _explicateur_factice()

    resultat = calculer_priorites_parc(df_historique, predicteur, explicateur)

    assert list(resultat["id_machine"]) == ["M1"]
    assert list(resultat.columns) == COLONNES_PRIORITES


def test_calculer_priorites_parc_trie_par_risque_decroissant():
    m1 = _historique_machine("M1", "L", heures_finales=5000.0)
    m2 = _historique_machine("M2", "M", heures_finales=8000.0)
    df_historique = pd.concat([m1, m2], ignore_index=True)

    proba_par_machine = {
        round(m1["heures_depuis_changement_outil"].iloc[-1], 4): 0.5,
        round(m2["heures_depuis_changement_outil"].iloc[-1], 4): 0.9,
    }
    predicteur = _predicteur_factice(seuil=0.1, proba_par_machine=proba_par_machine)
    explicateur = _explicateur_factice()

    resultat = calculer_priorites_parc(df_historique, predicteur, explicateur)

    assert list(resultat["id_machine"]) == ["M2", "M1"]


def test_calculer_priorites_parc_fenetre_qualitative_pas_dheure_precise():
    m1 = _historique_machine("M1", "L")
    proba = round(m1["heures_depuis_changement_outil"].iloc[-1], 4)
    predicteur = _predicteur_factice(seuil=0.1, proba_par_machine={proba: 0.9})
    explicateur = _explicateur_factice()

    resultat = calculer_priorites_parc(m1, predicteur, explicateur)

    fenetre = resultat.iloc[0]["fenetre"]
    assert "Risque élevé" in fenetre
    # Aucune heure précise (ex. "37h", "14:32"), seulement une fenêtre qualitative en jours.
    assert not re.search(r"\d+\s*h\b|\d+:\d+", fenetre)


def test_calculer_priorites_parc_aucune_alerte_retourne_dataframe_vide():
    m1 = _historique_machine("M1", "L")
    predicteur = _predicteur_factice(seuil=0.99, proba_par_machine={})
    explicateur = _explicateur_factice()

    resultat = calculer_priorites_parc(m1, predicteur, explicateur)

    assert resultat.empty
    assert list(resultat.columns) == COLONNES_PRIORITES
