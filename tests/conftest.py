import numpy as np
import pandas as pd
import pytest

from src.features import calculer_features
from src.predicteur import Predicteur

MESURES_SAINES = {
    "temperature_ambiante": 31.0,
    "temperature_process": 55.0,
    "vitesse_rotation": 1500.0,
    "couple": 30.0,
    "niveau_activite": 0.85,
    "vibration": 1.0,
    "humidite": 60.0,
    "concentration_poussiere": 35.0,
    "precipitation_mm": 0.0,
    "indicateur_coupure": 0,
    "niveau_tension": 207.0,
    "heures_depuis_changement_outil": 800.0,
    "heures_depuis_maintenance": 300.0,
}


def serie_machine(type_machine: str = "M", n: int = 60, **mesures) -> pd.DataFrame:
    """Relevés bruts d'une machine sur n pas de 30 min ; chaque mesure est un scalaire ou un tableau de longueur n."""
    return pd.DataFrame(
        {
            "id_machine": f"M_{type_machine}",
            "type_machine": type_machine,
            "timestamp": pd.date_range("2025-03-01", periods=n, freq="30min"),
            **(MESURES_SAINES | mesures),
            "panne": 0,
            "mode_panne": "",
        }
    )


@pytest.fixture(scope="session")
def predicteur():
    return Predicteur(utiliser_mlflow=True)


@pytest.fixture
def machine_normale() -> pd.DataFrame:
    """Machine H avec un outil récent, aucune dégradation."""
    return calculer_features(serie_machine("H")).iloc[[-1]]


@pytest.fixture
def machine_degradee() -> pd.DataFrame:
    """Machine L avec un outil en fin de vie : vibration et température en hausse."""
    serie = serie_machine(
        "L",
        heures_depuis_changement_outil=7000.0,
        heures_depuis_maintenance=2500.0,
        vibration=np.linspace(2.5, 4.5, 60),
        temperature_process=np.linspace(62.0, 72.0, 60),
        couple=48.0,
    )
    return calculer_features(serie).iloc[[-1]]
