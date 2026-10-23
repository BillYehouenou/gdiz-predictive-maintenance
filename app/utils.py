import logging

import duckdb
import mlflow
import pandas as pd
from fastapi import BackgroundTasks

from src.config import CHEMIN_DB, PAS, TABLE
from src.features import LIGNES_HISTORIQUE, calculer_features
from src.predicteur import Predicteur

logger = logging.getLogger(__name__)

predicteur = Predicteur(utiliser_mlflow=True)


def recharger_modele() -> dict:
    """Recharge le @champion courant sans redémarrer l'API."""
    global predicteur
    predicteur = Predicteur(utiliser_mlflow=True)
    logger.info(f"Modèle rechargé : version={predicteur.version}, seuil={predicteur.threshold:.4f}")
    return {"model_version": predicteur.version, "threshold": predicteur.threshold}


def _historique_machine(id_machine: str) -> pd.DataFrame:
    """Les dernières 24h de la machine, nécessaires aux features glissantes. Vide si indisponible."""
    try:
        with duckdb.connect(str(CHEMIN_DB), read_only=True) as conn:
            return conn.execute(
                f"SELECT * FROM {TABLE} WHERE id_machine = ? ORDER BY timestamp DESC LIMIT ?",
                [id_machine, LIGNES_HISTORIQUE],
            ).fetchdf()
    except Exception as e:
        logger.warning(f"Historique indisponible pour {id_machine} : {e}")
        return pd.DataFrame()


def _loguer_mlflow(id_machine: str, ligne: dict, probability: float, prediction: int) -> None:
    """Une prédiction = un run dans l'expérience de monitoring (lue par scripts/check_drift.py)."""
    try:
        mlflow.set_experiment("GDIZ_Monitoring_Predictions")
        niveau = "high" if probability >= predicteur.threshold * 3 else ("medium" if prediction else "normal")
        colonnes = ["heures_depuis_changement_outil", "vibration", "temperature_process", "concentration_poussiere"]
        colonnes += ["vibration_max_24h", "temperature_process_max_24h"]
        with mlflow.start_run(run_name=id_machine, tags={"id_machine": id_machine, "niveau_alerte": niveau}):
            mlflow.log_metrics(
                {"probabilite_panne": probability, "prediction": float(prediction)} | {c: float(ligne[c]) for c in colonnes}
            )
    except Exception as e:
        logger.warning(f"Log MLflow de monitoring échoué : {e}")


def traiter_prediction(donnees: dict, background_tasks: BackgroundTasks | None = None) -> dict:
    """
    Ajoute le relevé courant à l'historique de la machine (si id_machine est fourni), calcule les features
    avec la même fonction qu'à l'entraînement, puis prédit sur la dernière ligne.
    """
    id_machine = donnees.pop("id_machine", None)
    historique = _historique_machine(id_machine) if id_machine else pd.DataFrame()
    horodatage = historique["timestamp"].max() + PAS if not historique.empty else pd.Timestamp.now().floor(PAS)
    courant = pd.DataFrame([{**donnees, "id_machine": id_machine or "inconnue", "timestamp": horodatage}])
    serie = pd.concat([historique, courant], ignore_index=True) if not historique.empty else courant
    ligne = calculer_features(serie).iloc[[-1]]

    resultat = predicteur.predire(ligne).iloc[0]
    prediction, probabilite = int(resultat["prediction"]), float(resultat["probabilite_panne"])

    args = (id_machine or "inconnue", ligne.iloc[0].to_dict(), probabilite, prediction)
    if background_tasks is not None:
        background_tasks.add_task(_loguer_mlflow, *args)
    else:
        _loguer_mlflow(*args)
    return {"prediction": prediction, "probabilite_panne": round(probabilite, 4)}
