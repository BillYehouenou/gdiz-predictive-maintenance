from fastapi import APIRouter, FastAPI, HTTPException

from app.schemas import MachineData, PredictionResponse
from app.utils import process_prediction

app = FastAPI(
    title="GDIZ Maintenance Prédictive API",
    description="API de maintenance prédictive pour les machines textiles GDIZ Bénin",
    version="1.0.0",
)

router = APIRouter(prefix="/api/v1")


@app.get("/health")
def health():
    """Vérifie que l'API répond correctement."""
    return {"status": "healthy"}


@router.post("/predict", response_model=PredictionResponse)
def predict(data: MachineData):
    """Reçoit les données machine, valide le format et renvoie la prédiction."""
    try:
        result = process_prediction(data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)
