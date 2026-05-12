import pytest
import pandas as pd
import numpy as np
from src.xpredict import Predictor 

# Fixtures: données réutilisables pour les tests

@pytest.fixture
def predictor():
    """Initialise le prédicteur une seule fois pour tous les tests."""
    return Predictor(use_mlflow=True)

@pytest.fixture
def machine_normale():
    """Simule une machine qui va bien (faible température, faible usure)."""
    return pd.DataFrame({
        'ambient_temperature': [25.0], 'process_temperature': [35.0],
        'rotational_speed': [1500], 'torque': [40.0], 'tool_wear': [5],
        'machine_load': [0.5], 'vibration': [1.0], 'humidity': [50.0],
        'dust': [0.1], 'rain_flag': [0], 'power_outage': [0],
        'voltage': [220.0], 'voltage_stability': [0.99],
        'season': [0], 'machine_type': ['H']
    })

@pytest.fixture
def machine_en_surchauffe():
    """Simule une machine en train de rendre l'âme (température et usure critiques)."""
    return pd.DataFrame({
        'ambient_temperature': [45.0], 'process_temperature': [95.0],
        'rotational_speed': [2800], 'torque': [90.0], 'tool_wear': [240],
        'machine_load': [0.95], 'vibration': [8.5], 'humidity': [85.0],
        'dust': [0.8], 'rain_flag': [0], 'power_outage': [0],
        'voltage': [190.0], 'voltage_stability': [0.70],
        'season': [1], 'machine_type': ['L']
    })

# Batterie de tests

def test_chargement_modele(predictor):
    """Vérifie que le modèle est bien chargé en mémoire."""
    assert predictor.pipeline is not None
    assert hasattr(predictor.pipeline, "predict")

def test_format_sortie(predictor, machine_normale):
    """Vérifie que la sortie respecte le contrat de données."""
    res = predictor.predict(machine_normale)
    
    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {'prediction', 'failure_probability'}
    assert res['prediction'].dtype in [np.int64, np.int32, int]
    assert 0 <= res['failure_probability'].iloc[0] <= 1

def test_robustesse_colonnes_absentes(predictor):
    """Vérifie que le pipeline lève une erreur si une colonne cruciale manque."""
    data_incomplète = pd.DataFrame({'temp': [25]}) # Pas le bon nom de colonne
    with pytest.raises(Exception): # Scikit-learn doit lever une KeyError ou ValueError
        predictor.predict(data_incomplète)

def test_sensibilite_surchauffe(predictor, machine_en_surchauffe):
    """Test de 'non-régression' : vérifie que le modèle détecte bien un cas de panne évident.
    """
    res = predictor.predict(machine_en_surchauffe)
    # On s'attend à ce que la probabilité de panne soit élevée (> 0.5)
    assert res['failure_probability'].iloc[0] > 0.5
    assert res['prediction'].iloc[0] == 1

def test_stabilite_normal(predictor, machine_normale):
    """Vérifie qu'une machine saine est bien prédite comme saine."""
    res = predictor.predict(machine_normale)
    assert res['prediction'].iloc[0] == 0
    assert res['failure_probability'].iloc[0] < 0.3