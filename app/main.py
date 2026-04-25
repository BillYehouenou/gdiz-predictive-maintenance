from fastapi import FastAPI
from app.schemas import MachineData
from app.utils import load_model, make_prediction

app = FastAPI()

# Chargement du modèle au démarrage
model = load_model()

@app.get("/")
def read_root():
    return {"message": "API de Maintenance Prédictive avec IA opérationnelle"}

@app.post("/predict")
def predict(data: MachineData):
    # On transforme les données d'entrée en dictionnaire
    input_data = data.model_dump()
    
    if model:
        prediction, proba = make_prediction(model, input_data)
        status = "Danger : Panne imminente" if prediction == 1 else "Normal"
        return {
            "prediction": prediction,
            "confidence": round(proba, 2),
            "status": status
        }
    else:
        return {"error": "Modèle non disponible"}