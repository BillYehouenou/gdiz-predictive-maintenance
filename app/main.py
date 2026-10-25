import logging
from collections import Counter

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
from src.config import JOURS_PREDICTIONS, PAS, RACINE, TABLE
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


class FichiersStatiques(StaticFiles):
    """Le navigateur revalide chaque fichier (ETag) : une mise à jour n'est jamais mêlée à d'anciens scripts en cache."""

    def file_response(self, *args, **kwargs):
        reponse = super().file_response(*args, **kwargs)
        reponse.headers["Cache-Control"] = "no-cache"
        return reponse


app.mount("/static", FichiersStatiques(directory=RACINE / "web" / "static"), name="static")
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


def _historique(conn: duckdb.DuckDBPyConnection, ids_machines: list[str]) -> pd.DataFrame:
    """Les dernières 24 h de chaque machine demandée, en une requête."""
    if not ids_machines:
        return pd.DataFrame()
    return conn.execute(
        f"""SELECT * FROM {TABLE} WHERE id_machine IN (SELECT UNNEST(?))
            QUALIFY ROW_NUMBER() OVER (PARTITION BY id_machine ORDER BY timestamp DESC) <= ?""",
        [ids_machines, LIGNES_HISTORIQUE],
    ).fetchdf()


def _predire(releves: list[dict]) -> list[dict]:
    """
    Chaque relevé est ajouté à l'historique de sa machine (si connue), séparément des autres relevés du lot ;
    features calculées comme à l'entraînement, puis une seule prédiction pour tout le lot.
    """
    ids = sorted({r["id_machine"] for r in releves if r.get("id_machine")})
    historiques = dict(tuple(_lire(_historique, ids).groupby("id_machine"))) if ids else {}
    series = []
    for k, releve in enumerate(releves):
        historique = historiques.get(releve.get("id_machine"), pd.DataFrame())
        horodatage = historique["timestamp"].max() + PAS if not historique.empty else pd.Timestamp.now().floor(PAS)
        courant = pd.DataFrame([{**releve, "id_machine": releve.get("id_machine") or "inconnue", "timestamp": horodatage}])
        serie = pd.concat([historique, courant], ignore_index=True) if not historique.empty else courant
        series.append(serie.assign(id_machine=f"{k:06d}"))  # une série par relevé, dans l'ordre du lot
    features = calculer_features(pd.concat(series, ignore_index=True)).groupby("id_machine").tail(1)
    resultats = predicteur.predire(features)
    return [
        {"prediction": int(p), "probabilite_panne": round(float(r), 4)}
        for p, r in zip(resultats["prediction"], resultats["probabilite_panne"])
    ]


@app.get("/health")
def sante():
    return {"status": "healthy"}


@router.post("/predict", response_model=ReponsePrediction)
def predire(data: DonneesMachine):
    """Prédiction sur un relevé ; avec id_machine, les 24h d'historique en base complètent les features glissantes."""
    try:
        return _predire([data.model_dump()])[0]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")


@router.post("/predict/batch", response_model=list[ReponsePrediction])
def predire_lot(donnees: list[DonneesMachine]):
    """Prédiction sur un lot de relevés : une requête d'historique et un appel au modèle pour tout le lot."""
    if not donnees:
        return []
    try:
        return _predire([d.model_dump() for d in donnees])
    except HTTPException:
        raise
    except Exception:
        logger.exception("Erreur lors de la prédiction par lot")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")


@router.get("/parc", response_model=TableauParc)
def parc(periode: metier.CodePeriode = "annee"):
    """Indicateurs de la période et de la précédente, pannes par gamme et cause, coût des pannes par mois ou semaine."""
    return _lire(metier.tableau_parc, periode)


def _etat_parc(conn: duckdb.DuckDBPyConnection) -> dict:
    machines = metier.etat_parc(conn, predicteur.threshold)
    comptes = dict(Counter(m["statut"] for m in machines))
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
    derniere = calculer_features(_historique(conn, [id_machine])).iloc[[-1]]
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
def risque(id_machine: str, jours: int = Query(JOURS_PREDICTIONS, ge=1, le=JOURS_PREDICTIONS)):
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


# Pages du tableau de bord : le HTML ne contient que les libellés des périodes, le JavaScript lit l'API
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page_parc(request: Request):
    periodes = metier.libelles_periodes(_lire(metier.fin_donnees))
    return gabarits.TemplateResponse(request, "parc.html", {"titre": "Parc", "page": "parc", "periodes": periodes})


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
