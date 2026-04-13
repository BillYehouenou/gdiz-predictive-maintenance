from app.schemas import MachineData

def calculate_features(data: MachineData):
    # 1. Calcul du delta de température (Feature Engineering)
    temp_diff = data.process_temperature - data.air_temperature
    
    # 2. Calcul d'un ratio puissance (Couple * Vitesse) 
    # Juste pour l'exemple, c'est souvent utile en maintenance
    power_factor = data.torque * data.rotational_speed
    
    # On retourne un dictionnaire de features "propres"
    return {
        "temp_diff": temp_diff,
        "power_factor": power_factor,
        "tool_wear": data.tool_wear
    }

def simple_heuristic_model(features: dict):
    # On simule un modèle qui prend les features calculées
    # Si la différence de température > 15 ET l'usure > 180 min -> Panne
    if features["temp_diff"] > 15 and features["tool_wear"] > 180:
        return 1, 0.85 # (Classe 1, Probabilité 85%)
    
    return 0, 0.15 # (Classe 0, Probabilité 15%)