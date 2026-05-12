from fastapi import FastAPI, HTTPException
from app.schemas import MachineData, PredictionResponse
from app.utils import process_prediction

# Pour lancer : uv run uvicorn app.main:app --reload
app = FastAPI()

@app.get("/")
def root():
    """Vérifie si l'API est bien en ligne."""
    return {"status": "ok", "message": "GDIZ Maintenance API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: MachineData):
    """Reçoit les données machine, valide le format et renvoie la prédiction.
    """
    try:
        input_data = data.model_dump()
        result = process_prediction(input_data)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))