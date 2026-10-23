"""
Entraînement : LightGBM --> calibration sigmoïde et seuil choisis sur la validation --> métriques sur le test
--> MLflow. Le @champion n'est remplacé que si le recall événementiel ne régresse pas.
Usage : python -m src.train
"""

import json
import logging
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import CIBLE, DOSSIER_MODELES, NOM_EXPERIENCE, NOM_MODELE, PAS, URI_MLFLOW
from src.donnees import DATE_TEST, DATE_VALIDATION, HORIZON, MACHINES_RESERVEES, charger, decouper
from src.evaluation import grille_seuils, metriques_evenementielles, trouver_seuil_pour_budget_fa
from src.features import preparer_X

logger = logging.getLogger(__name__)

ALIAS_CHAMPION = "champion"
METRIQUE_PROMOTION = "recall_evenementiel"
PAS_HORIZON = int(HORIZON / PAS)
BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR = 0.2  # dérangements inutiles acceptés par l'équipe de maintenance

# Peu de feuilles et forte régularisation : le signal de panne est rare
HYPERPARAMETRES: dict[str, Any] = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 80,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 0.3,
}


def entrainer(df: pd.DataFrame) -> tuple[CalibratedClassifierCV, float, dict]:
    train, validation, test = decouper(df)
    logger.info(f"Train {len(train):,} | validation {len(validation):,} | test {len(test):,} lignes")

    modele = LGBMClassifier(**HYPERPARAMETRES, random_state=42, n_jobs=-1, verbose=-1)
    modele.fit(preparer_X(train), train[CIBLE])
    # Sigmoïde plutôt qu'isotonique : des probabilités continues permettent de régler finement le seuil
    calibre = CalibratedClassifierCV(FrozenEstimator(modele), method="sigmoid").fit(preparer_X(validation), validation[CIBLE])

    probas_validation = calibre.predict_proba(preparer_X(validation))[:, 1]
    seuil, _ = trouver_seuil_pour_budget_fa(
        validation, probas_validation, grille_seuils(probas_validation), PAS_HORIZON, BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR
    )

    probas_brutes = modele.predict_proba(preparer_X(test))[:, 1]
    probas = calibre.predict_proba(preparer_X(test))[:, 1]
    evenements = metriques_evenementielles(test, probas, seuil, PAS_HORIZON)
    reservees = test["id_machine"].isin(MACHINES_RESERVEES).to_numpy()
    evenements_reservees = metriques_evenementielles(test[reservees], probas[reservees], seuil, PAS_HORIZON)

    metriques = {
        "recall_evenementiel": evenements["recall_evenementiel"],
        "taux_fausses_alertes_par_machine_jour": evenements["taux_fausses_alertes_par_machine_jour"],
        "n_failures_test": evenements["nb_pannes"],
        "n_caught_test": evenements["nb_attrapees"],
        "preavis_median_h": float(np.median(evenements["preavis_heures"])) if evenements["preavis_heures"] else 0.0,
        "recall_machines_reservees": evenements_reservees["recall_evenementiel"],
        "pr_auc": average_precision_score(test[CIBLE], probas),
        "roc_auc": roc_auc_score(test[CIBLE], probas),
        "brier_score_raw": brier_score_loss(test[CIBLE], probas_brutes),
        "brier_score_calibrated": brier_score_loss(test[CIBLE], probas),
    }
    return calibre, seuil, metriques


def promouvoir_si_meilleur(client: mlflow.MlflowClient, version: str, score: float) -> bool:
    """Promeut la version en @champion si elle égale ou dépasse le champion actuel (automatique s'il n'y en a pas)."""
    try:
        champion = client.get_model_version_by_alias(NOM_MODELE, ALIAS_CHAMPION)
        score_champion = client.get_run(str(champion.run_id)).data.metrics.get(METRIQUE_PROMOTION, -1.0)
    except Exception:
        score_champion = -1.0

    if score < score_champion:
        logger.warning(f"Version {version} ({score:.4f}) sous le champion ({score_champion:.4f}) : champion conservé.")
        return False
    client.set_registered_model_alias(NOM_MODELE, ALIAS_CHAMPION, version)
    logger.info(f"Version {version} promue @{ALIAS_CHAMPION} ({score:.4f} >= {score_champion:.4f}).")
    return True


def main():
    mlflow.set_tracking_uri(URI_MLFLOW)
    mlflow.set_experiment(NOM_EXPERIENCE)
    calibre, seuil, metriques = entrainer(charger())

    with mlflow.start_run(run_name="LGBM_Training"):
        mlflow.log_params(
            {
                **HYPERPARAMETRES,
                "seuil_optimal": round(seuil, 4),
                "budget_fa_machine_jour": BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR,
                "horizon_heures": HORIZON / pd.Timedelta("1h"),
                "date_validation": str(DATE_VALIDATION.date()),
                "date_test": str(DATE_TEST.date()),
            }
        )
        mlflow.log_metrics(metriques)
        info = mlflow.sklearn.log_model(sk_model=calibre, name="model_pipeline", registered_model_name=NOM_MODELE)

    if promouvoir_si_meilleur(mlflow.MlflowClient(), info.registered_model_version, metriques[METRIQUE_PROMOTION]):
        DOSSIER_MODELES.mkdir(exist_ok=True)
        joblib.dump(calibre, DOSSIER_MODELES / "model_pipeline.pkl")
        (DOSSIER_MODELES / "seuil_optimal.json").write_text(json.dumps({"threshold": seuil}))

    for nom, valeur in metriques.items():
        print(f"{nom}: {valeur:.4f}" if isinstance(valeur, float) else f"{nom}: {valeur}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
