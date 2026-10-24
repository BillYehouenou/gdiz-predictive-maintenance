import logging

import duckdb
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import (
    DonneesMachine,
    EtatParc,
    FicheMachine,
    InfoModele,
    PointRisque,
    ReponsePrediction,
    ReponseRechargement,
    TableauParc,
)
from src import metier
from src.config import PAS, RACINE, TABLE
from src.explain import ExplicateurCause
from src.features import LIGNES_HISTORIQUE, calculer_features
from src.predicteur import Predicteur

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GDIZ Maintenance Prédictive API",
    description="Maintenance prédictive des 50 machines textiles : prédiction, état du parc, fiche machine",
    version="4.0.0",
)
router = APIRouter(prefix="/api/v1")
app.mount("/static", StaticFiles(directory=RACINE / "web" / "static"), name="static")
gabarits = Jinja2Templates(directory=RACINE / "web" / "templates")
predicteur = Predicteur(utiliser_mlflow=True)
explicateur = ExplicateurCause(predicteur.modele)


def _lire(fonction, *args):
    """Exécute une lecture sur la base principale ; 503 si les tables du pipeline manquent."""
    try:
        with metier.connexion() as conn:
            return fonction(conn, *args)
    except duckdb.CatalogException:
        raise HTTPException(status_code=503, detail="Données incomplètes : lancer `make pipeline`.")


def _historique(conn: duckdb.DuckDBPyConnection, id_machine: str) -> pd.DataFrame:
    return conn.execute(
        f"SELECT * FROM {TABLE} WHERE id_machine = ? ORDER BY timestamp DESC LIMIT ?", [id_machine, LIGNES_HISTORIQUE]
    ).fetchdf()


def _predire(donnees: dict) -> dict:
    """Ajoute le relevé à l'historique de la machine (si connue), calcule les features comme à l'entraînement, prédit."""
    id_machine = donnees.pop("id_machine", None)
    historique = _lire(_historique, id_machine) if id_machine else pd.DataFrame()
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")


@router.post("/predict/batch", response_model=list[ReponsePrediction])
def predire_lot(donnees: list[DonneesMachine]):
    return [predire(d) for d in donnees]


@router.get("/parc", response_model=TableauParc)
def parc(periode: str = Query("annee", pattern="^(30j|t4|s2|annee|tout)$")):
    """Indicateurs de la période et de la précédente, pannes par gamme et cause, coût des pannes par mois ou semaine."""
    return _lire(metier.tableau_parc, periode)


def _etat_parc(conn: duckdb.DuckDBPyConnection) -> dict:
    machines = metier.etat_parc(conn, predicteur.threshold)
    comptes = pd.Series([m["statut"] for m in machines]).value_counts().to_dict()
    return {
        "fin_donnees": metier.fin_donnees(conn),
        "seuil": predicteur.threshold,
        "machines": machines,
        "comptes": comptes,
    }


@router.get("/etat", response_model=EtatParc)
def etat():
    """Statut actuel des 50 machines : plan de l'atelier, matrice risque × criticité, compteur de la navigation."""
    return _lire(_etat_parc)


def _fiche(conn: duckdb.DuckDBPyConnection, id_machine: str) -> dict | None:
    fiche = metier.fiche_machine(conn, id_machine, predicteur.threshold)
    if fiche is None:
        return None
    derniere = calculer_features(_historique(conn, id_machine).iloc[::-1]).iloc[[-1]]
    fiche["contributions"] = [
        {**c, **dict(zip(("libelle", "unite"), metier.libelle_feature(c["feature"])))}
        for c in explicateur.contributions(derniere)
    ]
    return fiche


@router.get("/machines/{id_machine}", response_model=FicheMachine)
def machine(id_machine: str):
    """Fiche d'une machine : statut, santé, risque, contributions SHAP, historique, 12 mois, capteurs sur 30 jours."""
    fiche = _lire(_fiche, id_machine)
    if fiche is None:
        raise HTTPException(status_code=404, detail=f"Machine inconnue : {id_machine}")
    return fiche


@router.get("/machines/{id_machine}/risque", response_model=list[PointRisque])
def risque(id_machine: str, jours: int = Query(60, ge=1, le=730)):
    """Risque prédit par pas de 2 h, avec les délestages (pas de mesure) et les pannes constatées."""
    serie = _lire(metier.serie_risque, id_machine, jours)
    if serie.empty:
        raise HTTPException(status_code=404, detail=f"Machine inconnue : {id_machine}")
    return serie.to_dict("records")


@router.get("/info", response_model=InfoModele)
def info_modele():
    """Version et seuil du @champion chargé en mémoire."""
    return {"model_name": predicteur.nom_modele, "model_version": predicteur.version, "threshold": predicteur.threshold}


@router.post("/reload", response_model=ReponseRechargement)
def recharger():
    """Recharge le @champion sans redémarrer, après `make train score`."""
    global predicteur, explicateur
    try:
        predicteur = Predicteur(utiliser_mlflow=True)
        explicateur = ExplicateurCause(predicteur.modele)
    except Exception:
        logger.exception("Erreur lors du rechargement du modèle")
        raise HTTPException(status_code=500, detail="Erreur interne lors du rechargement du modèle.")
    return {"model_version": predicteur.version, "threshold": predicteur.threshold, "status": "success"}


app.include_router(router)


# Pages du tableau de bord : le HTML ne contient aucune donnée, le JavaScript lit l'API
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page_parc(request: Request):
    return gabarits.TemplateResponse(request, "parc.html", {"titre": "Parc", "page": "parc"})


@app.get("/machines", include_in_schema=False)
def page_machine_prioritaire():
    """Sans machine précisée : la plus à risque parmi les machines en marche."""
    machines = _lire(_etat_parc)["machines"]
    en_marche = [m for m in machines if m["statut"] != "en_panne"] or machines
    return RedirectResponse(f"/machines/{max(en_marche, key=lambda m: m['risque'])['id_machine']}")


@app.get("/machines/{id_machine}", response_class=HTMLResponse, include_in_schema=False)
def page_machine(request: Request, id_machine: str):
    if not id_machine.startswith("GDIZ_TEX_M_"):
        raise HTTPException(status_code=404, detail=f"Machine inconnue : {id_machine}")
    return gabarits.TemplateResponse(
        request, "machine.html", {"titre": f"Machine M{id_machine[-2:]}", "page": "machine", "id_machine": id_machine}
    )
