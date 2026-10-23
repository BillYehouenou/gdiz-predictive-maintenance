import logging

import duckdb
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query

from app.schemas import DonneesMachine, EtatMachine, InfoModele, PointSerie, ReponsePrediction, ReponseRechargement
from src.config import CHEMIN_DB, PAS, TABLE, TABLE_PREDICTIONS
from src.features import LIGNES_HISTORIQUE, calculer_features
from src.predicteur import Predicteur

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GDIZ Maintenance Prédictive API",
    description="API de maintenance prédictive pour les machines textiles GDIZ Bénin",
    version="3.1.0",
)
router = APIRouter(prefix="/api/v1")
predicteur = Predicteur(utiliser_mlflow=True)

_SQL_ETAT_PARC = f"""
    SELECT t.id_machine, t.type_machine, t.timestamp, p.probabilite_panne, p.niveau, p.cause_principale,
           t.heures_depuis_changement_outil, t.heures_depuis_maintenance
    FROM {TABLE} t JOIN {TABLE_PREDICTIONS} p USING (id_machine, timestamp)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY t.id_machine ORDER BY t.timestamp DESC) = 1
"""


def _lire(sql: str, parametres: list | None = None) -> pd.DataFrame:
    try:
        with duckdb.connect(str(CHEMIN_DB), read_only=True) as conn:
            return conn.execute(sql, parametres or []).fetchdf()
    except duckdb.CatalogException:
        raise HTTPException(status_code=503, detail="Table predictions absente : lancer `make score`.")


def _enregistrements(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _predire(donnees: dict) -> dict:
    """Ajoute le relevé à l'historique de la machine (si connue), calcule les features comme à l'entraînement, prédit."""
    id_machine = donnees.pop("id_machine", None)
    historique = (
        _lire(f"SELECT * FROM {TABLE} WHERE id_machine = ? ORDER BY timestamp DESC LIMIT ?", [id_machine, LIGNES_HISTORIQUE])
        if id_machine
        else pd.DataFrame()
    )
    horodatage = historique["timestamp"].max() + PAS if not historique.empty else pd.Timestamp.now().floor(PAS)
    courant = pd.DataFrame([{**donnees, "id_machine": id_machine or "inconnue", "timestamp": horodatage}])
    serie = pd.concat([historique, courant], ignore_index=True) if not historique.empty else courant
    resultat = predicteur.predire(calculer_features(serie).iloc[[-1]]).iloc[0]
    return {"prediction": int(resultat["prediction"]), "probabilite_panne": round(float(resultat["probabilite_panne"]), 4)}


@app.get("/health")
def sante():
    return {"status": "healthy"}


@router.post("/predict", response_model=ReponsePrediction)
def predire(data: DonneesMachine):
    """Prédiction sur un relevé ; avec id_machine, les 24h d'historique en base complètent les features glissantes."""
    try:
        return _predire(data.model_dump())
    except Exception:
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")


@router.post("/predict/batch", response_model=list[ReponsePrediction])
def predire_lot(donnees: list[DonneesMachine]):
    try:
        return [_predire(d.model_dump()) for d in donnees]
    except Exception:
        logger.exception("Erreur lors de la prédiction par lot")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction par lot.")


@router.get("/machines", response_model=list[EtatMachine])
def machines():
    """Dernier état de chaque machine."""
    return _enregistrements(_lire(f"{_SQL_ETAT_PARC} ORDER BY id_machine"))


@router.get("/priorites", response_model=list[EtatMachine])
def priorites():
    """Machines en alerte sur leur dernier relevé, de la plus à la moins risquée."""
    return _enregistrements(_lire(f"SELECT * FROM ({_SQL_ETAT_PARC}) WHERE niveau <> 'normal' ORDER BY probabilite_panne DESC"))


@router.get("/machines/{id_machine}/serie", response_model=list[PointSerie])
def serie_machine(id_machine: str, jours: int = Query(30, ge=1, le=730)):
    """Capteurs clés et risque prédit sur les derniers jours."""
    df = _lire(
        f"""SELECT t.timestamp, p.probabilite_panne, t.vibration, t.temperature_process, t.couple,
                   t.concentration_poussiere, t.panne
            FROM {TABLE} t JOIN {TABLE_PREDICTIONS} p USING (id_machine, timestamp)
            WHERE t.id_machine = ?
              AND t.timestamp > (SELECT MAX(timestamp) FROM {TABLE} WHERE id_machine = ?) - ? * INTERVAL 1 DAY
            ORDER BY t.timestamp""",
        [id_machine, id_machine, jours],
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Machine inconnue : {id_machine}")
    return _enregistrements(df)


@router.get("/info", response_model=InfoModele)
def info_modele():
    """Version et seuil du @champion chargé en mémoire."""
    return {"model_name": predicteur.nom_modele, "model_version": predicteur.version, "threshold": predicteur.threshold}


@router.post("/reload", response_model=ReponseRechargement)
def recharger():
    """Recharge le @champion sans redémarrer, après `make train score`."""
    global predicteur
    try:
        predicteur = Predicteur(utiliser_mlflow=True)
    except Exception:
        logger.exception("Erreur lors du rechargement du modèle")
        raise HTTPException(status_code=500, detail="Erreur interne lors du rechargement du modèle.")
    return {"model_version": predicteur.version, "threshold": predicteur.threshold, "status": "success"}


app.include_router(router)
