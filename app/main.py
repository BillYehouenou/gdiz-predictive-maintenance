import logging
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from app import utils
from app.schemas import DonneesMachine, InfoModele, ReponsePrediction, ReponseRechargement

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GDIZ Maintenance Prédictive API",
    description="API de maintenance prédictive pour les machines textiles GDIZ Bénin",
    version="2.0.0",
)

router = APIRouter(prefix="/api/v1")


@app.get("/health")
def sante():
    """Vérifie que l'API répond correctement."""
    return {"status": "healthy"}


@router.post("/predict", response_model=ReponsePrediction)
def predire(data: DonneesMachine, background_tasks: BackgroundTasks):
    """Reçoit les données machine, valide le format et renvoie la prédiction."""
    try:
        return utils.traiter_prediction(data.model_dump(), background_tasks=background_tasks)
    except Exception:
        logger.exception("Erreur lors du traitement de la prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")


@router.post("/predict/batch", response_model=list[ReponsePrediction])
def predire_lot(donnees: list[DonneesMachine], background_tasks: BackgroundTasks):
    """
    Prédit sur plusieurs machines en un seul appel.
    """
    try:
        return [utils.traiter_prediction(d.model_dump(), background_tasks=background_tasks) for d in donnees]
    except Exception:
        logger.exception("Erreur lors du traitement d'un lot de prédictions")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction par lot.")


@router.get("/info", response_model=InfoModele)
def info_modele():
    """Version et seuil du modèle @champion actuellement chargé en mémoire."""
    return {
        "model_name": utils.predicteur.nom_modele,
        "model_version": utils.predicteur.version,
        "threshold": utils.predicteur.threshold,
    }


@router.post("/reload", response_model=ReponseRechargement)
def recharger():
    """
    Recharge le champion `@champion` courant sans redémarrer le service.
    A appeler après un `make train` qui promeut une nouvelle version.
    """
    try:
        info = utils.recharger_modele()
        return {**info, "status": "success"}
    except Exception:
        logger.exception("Erreur lors du rechargement du modèle")
        raise HTTPException(status_code=500, detail="Erreur interne lors du rechargement du modèle.")


app.include_router(router)
