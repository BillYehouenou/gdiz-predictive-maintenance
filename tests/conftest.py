import pandas as pd
import pytest

from src.xpredict import Predicteur


@pytest.fixture(scope="session")
def predicteur():
    """Charge le prédicteur une seule fois pour toute la session de tests."""
    return Predicteur(utiliser_mlflow=True)


@pytest.fixture
def machine_normale():
    """Machine H en saison des pluies — état sain nominal, outil récent, aucune dégradation récente."""
    return pd.DataFrame(
        {
            "temperature_ambiante": [26.0],
            "temperature_process": [40.0],
            "vitesse_rotation": [1200.0],
            "couple": [28.0],
            "heures_depuis_changement_outil": [800.0],
            "heures_depuis_maintenance": [200.0],
            "niveau_activite": [0.5],
            "vibration": [1.2],
            "humidite": [70.0],
            "concentration_poussiere": [45.0],
            "niveau_tension": [225.0],
            "precipitation_mm": [2.0],
            "indicateur_coupure": [0],
            "type_machine": ["H"],
            # Features temporelles : machine stable, aucune variation sur 24h
            "vibration_max_24h": [1.3],
            "vibration_moyenne_24h": [1.2],
            "vibration_ecart_type_24h": [0.1],
            "vibration_delta_4h": [0.05],
            "temperature_process_max_24h": [41.0],
            "temperature_process_moyenne_24h": [40.0],
            "temperature_process_delta_4h": [0.5],
            "ecart_thermique": [14.0],
            "ecart_thermique_max_24h": [15.0],
            "poussiere_moyenne_24h": [42.0],
            "poussiere_max_24h": [50.0],
            "couple_max_24h": [32.0],
        }
    )


@pytest.fixture
def machine_en_surchauffe():
    """Machine L en Harmattan — outil en fin de vie + délestage + dégradation rapide sur 24h."""
    return pd.DataFrame(
        {
            "temperature_ambiante": [41.0],
            "temperature_process": [89.0],
            "vitesse_rotation": [2500.0],
            "couple": [85.0],
            "heures_depuis_changement_outil": [9000.0],
            "heures_depuis_maintenance": [3000.0],
            "niveau_activite": [0.95],
            "vibration": [7.5],
            "humidite": [22.0],
            "concentration_poussiere": [380.0],
            "niveau_tension": [195.0],
            "precipitation_mm": [0.0],
            "indicateur_coupure": [1],
            "type_machine": ["L"],
            # Features temporelles : machine en dégradation rapide sur les 24h précédentes
            "vibration_max_24h": [8.2],
            "vibration_moyenne_24h": [6.5],
            "vibration_ecart_type_24h": [1.8],
            "vibration_delta_4h": [2.1],
            "temperature_process_max_24h": [92.0],
            "temperature_process_moyenne_24h": [85.0],
            "temperature_process_delta_4h": [6.0],
            "ecart_thermique": [48.0],
            "ecart_thermique_max_24h": [50.0],
            "poussiere_moyenne_24h": [350.0],
            "poussiere_max_24h": [410.0],
            "couple_max_24h": [88.0],
        }
    )
