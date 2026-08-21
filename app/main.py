import logging

from fastapi import APIRouter, FastAPI, HTTPException

from app.schemas import DonneesMachine, ReponsePrediction
from app.utils import traiter_prediction

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GDIZ Maintenance Prédictive API",
    description="API de maintenance prédictive pour les machines textiles GDIZ Bénin",
    version="1.0.0",
)

router = APIRouter(prefix="/api/v1")


@app.get("/health")
def sante():
    """Vérifie que l'API répond correctement."""
    return {"status": "healthy"}


@router.post("/predict", response_model=ReponsePrediction)
def predire(data: DonneesMachine):
    """Reçoit les données machine, valide le format et renvoie la prédiction."""
    try:
        result = traiter_prediction(data.model_dump())
        return result
    except Exception:
        logger.exception("Erreur lors du traitement de la prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")


app.include_router(router)
