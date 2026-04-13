from fastapi import FastAPI
from app.schemas import MachineData, PredictionResponse
from app.utils import calculate_features, simple_heuristic_model
import random # On simule le modèle pour l'instant

app = FastAPI(title="Maintenance Prédictive API")

@app.get("/")
def read_root():
    return {"message": "API de Maintenance Prédictive opérationnelle"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: MachineData):
    # 1. On transforme les données brutes en features
    features = calculate_features(data)
    
    # 2. On passe les features au modèle
    prediction, probability = simple_heuristic_model(features)
    
    # 3. Logique pour le statut (le "Label")
    status = "Danger : Maintenance requise" if prediction == 1 else "Normal"
    
    return {
        "prediction": prediction,
        "probability": probability,
        "status": status
    }