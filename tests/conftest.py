import pandas as pd
import pytest
from src.xpredict import Predictor


@pytest.fixture(scope="session")
def predictor():
    """Charge le prédicteur une seule fois pour toute la session de tests."""
    return Predictor(use_mlflow=True)


@pytest.fixture
def machine_normale():
    """Machine H en saison des pluies — état sain nominal."""
    return pd.DataFrame(
        {
            "ambient_temperature": [26.0],
            "process_temperature": [40.0],
            "rotational_speed": [1200.0],
            "torque": [28.0],
            "tool_wear": [20.0],
            "activity_level": [0.5],
            "vibration": [1.2],
            "humidity": [70.0],
            "dust_concentration": [45.0],
            "voltage_level": [225.0],
            "voltage_stability": [82.0],
            "rain_flag": [1],
            "power_loss_indicator": [0],
            "benin_season": [2],
            "machine_type": ["H"],
        }
    )


@pytest.fixture
def machine_en_surchauffe():
    """Machine L en Harmattan — usure critique + délestage + haute poussière."""
    return pd.DataFrame(
        {
            "ambient_temperature": [41.0],
            "process_temperature": [89.0],
            "rotational_speed": [2500.0],
            "torque": [85.0],
            "tool_wear": [90.0],
            "activity_level": [0.95],
            "vibration": [7.5],
            "humidity": [22.0],
            "dust_concentration": [380.0],
            "voltage_level": [195.0],
            "voltage_stability": [18.0],
            "rain_flag": [0],
            "power_loss_indicator": [1],
            "benin_season": [1],
            "machine_type": ["L"],
        }
    )
