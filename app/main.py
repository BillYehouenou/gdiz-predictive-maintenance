"""API et tableau de bord sur le même port : prédiction, état du parc, fiche machine, pages web."""

import logging
from collections import Counter

import duckdb
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import DonneesMachine, EtatParc, FicheMachine, InfoModele, PointRisque, ReponsePrediction, TableauParc
from src import metier
from src.config import JOURS_PREDICTIONS, PAS, RACINE, TABLE
from src.donnees import LIGNES_HISTORIQUE, calculer_features
from src.modele import ExplicateurCause, Predicteur

logger = logging.getLogger(__name__)
app = FastAPI(title="GDIZ Maintenance Prédictive API", description="Prédiction, état du parc et fiche des 50 machines textiles")
router = APIRouter(prefix="/api/v1")


class FichiersStatiques(StaticFiles):
    """Le navigateur revalide chaque fichier (ETag) : une mise à jour n'est jamais mêlée à d'anciens scripts en cache."""

    def file_response(self, *args, **kwargs):
        reponse = super().file_response(*args, **kwargs)
        reponse.headers["Cache-Control"] = "no-cache"
        return reponse


app.mount("/static", FichiersStatiques(directory=RACINE / "web" / "static"), name="static")
gabarits = Jinja2Templates(directory=RACINE / "web" / "templates")
predicteur = Predicteur()
explicateur = ExplicateurCause(predicteur.modele)


def _lire(fonction, *args):
    """Exécute une lecture sur la base ; 503 si les tables du pipeline manquent."""
    try:
        with metier.connexion() as conn:
            return fonction(conn, *args)
    except duckdb.CatalogException:
        raise HTTPException(status_code=503, detail="Données incomplètes : lancer `make pipeline`.")


def _historique(conn: duckdb.DuckDBPyConnection, ids_machines: list[str]) -> pd.DataFrame:
    """Les dernières 24 h de chaque machine demandée, en une requête."""
    return conn.execute(
        f"""SELECT * FROM {TABLE} WHERE id_machine IN (SELECT UNNEST(?))
            QUALIFY ROW_NUMBER() OVER (PARTITION BY id_machine ORDER BY timestamp DESC) <= ?""",
        [ids_machines, LIGNES_HISTORIQUE],
    ).fetchdf()


def _predire(releves: list[dict]) -> list[dict]:
    """
    Chaque relevé est ajouté à l'historique de sa machine (si connue), séparément des autres relevés du lot ; features
    calculées comme à l'entraînement, puis une seule prédiction pour tout le lot.
    """
    try:
        ids = sorted({r["id_machine"] for r in releves if r.get("id_machine")})
        historiques = dict(tuple(_lire(_historique, ids).groupby("id_machine"))) if ids else {}
        series = []
        for k, releve in enumerate(releves):
            historique = historiques.get(releve.get("id_machine"), pd.DataFrame())
            horodatage = historique["timestamp"].max() + PAS if not historique.empty else pd.Timestamp.now().floor(PAS)
            courant = pd.DataFrame([{**releve, "timestamp": horodatage}])
            serie = pd.concat([historique, courant], ignore_index=True) if not historique.empty else courant
            series.append(serie.assign(id_machine=f"{k:06d}"))  # une série par relevé, dans l'ordre du lot
        resultats = predicteur.predire(calculer_features(pd.concat(series, ignore_index=True)).groupby("id_machine").tail(1))
        return [{"prediction": int(p), "probabilite_panne": round(float(r), 4)} for p, r in resultats.to_numpy()]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")


@app.get("/health")
def sante():
    return {"status": "healthy"}


@router.post("/predict", response_model=ReponsePrediction)
def predire(data: DonneesMachine):
    return _predire([data.model_dump()])[0]


@router.post("/predict/batch", response_model=list[ReponsePrediction])
def predire_lot(donnees: list[DonneesMachine]):
    """Une requête d'historique et un appel au modèle pour tout le lot."""
    return _predire([d.model_dump() for d in donnees]) if donnees else []


@router.get("/parc", response_model=TableauParc)
def parc(periode: metier.CodePeriode = "annee"):
    """Indicateurs de la période et de la précédente, pannes par gamme et cause, coût par mois ou semaine."""
    return _lire(metier.tableau_parc, periode)


def _etat(conn: duckdb.DuckDBPyConnection) -> dict:
    machines = metier.etat_parc(conn, predicteur.threshold)
    return {"fin_donnees": metier.fin_donnees(conn), "seuil": predicteur.threshold, "machines": machines,
            "comptes": dict(Counter(m["statut"] for m in machines))}  # fmt: skip


@router.get("/etat", response_model=EtatParc)
def etat():
    """Statut actuel des 50 machines : plan de l'atelier et matrice risque × criticité."""
    return _lire(_etat)


def _fiche(conn: duckdb.DuckDBPyConnection, id_machine: str) -> dict:
    fiche = metier.fiche_machine(conn, id_machine, predicteur.threshold)
    if fiche is None:
        raise HTTPException(status_code=404, detail=f"Machine inconnue : {id_machine}")
    features = calculer_features(_historique(conn, [id_machine]))
    en_marche = features[features["etat"] == "marche"]  # le modèle n'explique que des relevés en marche
    fiche["contributions"] = [
        {**c, **dict(zip(("libelle", "unite"), metier.libelle_feature(c["feature"])))}
        for c in explicateur.contributions((en_marche if not en_marche.empty else features).iloc[[-1]])
    ]
    return fiche


@router.get("/machines/{id_machine}", response_model=FicheMachine)
def machine(id_machine: str):
    """Statut, santé, risque, contributions SHAP, historique, 12 mois, capteurs sur 30 jours."""
    return _lire(_fiche, id_machine)


@router.get("/machines/{id_machine}/risque", response_model=list[PointRisque])
def risque(id_machine: str, jours: int = Query(JOURS_PREDICTIONS, ge=1, le=JOURS_PREDICTIONS)):
    """Risque prédit par pas de 2 h, délestages et pannes constatées."""
    serie = _lire(metier.serie_risque, id_machine, jours)
    if serie.empty:
        raise HTTPException(status_code=404, detail=f"Machine inconnue : {id_machine}")
    return serie.astype(object).where(serie.notna(), None).to_dict("records")


@router.get("/info", response_model=InfoModele)
def info_modele():
    """Version et seuil du @champion chargé en mémoire."""
    return {"model_name": predicteur.nom_modele, "model_version": predicteur.version, "threshold": predicteur.threshold}


@router.post("/reload", response_model=InfoModele)
def recharger():
    """Recharge le @champion sans redémarrer, après `make train score`."""
    global predicteur, explicateur
    try:
        predicteur = Predicteur()
        explicateur = ExplicateurCause(predicteur.modele)
    except Exception:
        logger.exception("Erreur lors du rechargement du modèle")
        raise HTTPException(status_code=500, detail="Erreur interne lors du rechargement du modèle.")
    return info_modele()


app.include_router(router)


# Pages : le HTML ne contient que les libellés des périodes, le JavaScript lit l'API
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page_parc(request: Request):
    periodes = metier.libelles_periodes(_lire(metier.fin_donnees))
    return gabarits.TemplateResponse(request, "parc.html", {"titre": "Parc", "page": "parc", "periodes": periodes})


@app.get("/machines", include_in_schema=False)
def page_machine_prioritaire():
    """Sans machine précisée : la plus à risque parmi celles qui ne sont pas en panne."""
    machines = _lire(_etat)["machines"]
    candidates = [m for m in machines if m["statut"] != "en_panne"] or machines
    return RedirectResponse(f"/machines/{max(candidates, key=lambda m: m['risque'])['id_machine']}")


@app.get("/machines/{id_machine}", response_class=HTMLResponse, include_in_schema=False)
def page_machine(request: Request, id_machine: str):
    if not id_machine.startswith("GDIZ_TEX_M_"):
        raise HTTPException(status_code=404, detail=f"Machine inconnue : {id_machine}")
    contexte = {"titre": f"Machine M{id_machine[-2:]}", "page": "machine", "id_machine": id_machine}
    return gabarits.TemplateResponse(request, "machine.html", contexte)
