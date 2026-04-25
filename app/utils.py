import joblib
import numpy as np
import os

# On définit le chemin du modèle
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")

def load_model():
    """Charge le modèle depuis le disque"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def make_prediction(model, data_dict):
    """Utilise le modèle pour prédire la panne"""
    # Variables d'entrée dans l'ordre attendu par le modèle
    features_list = [
        data_dict['air_temperature'],
        data_dict['process_temperature'],
        data_dict['rotational_speed'],
        data_dict['torque'],
        data_dict['tool_wear']
    ]
    # Convertir le dictionnaire en format attendu par Scikit-Learn (2D array)
    features = np.array([features_list])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()
    
    return int(prediction), float(probability)