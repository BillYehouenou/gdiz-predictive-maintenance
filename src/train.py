"""
Entraînement du LightGBM retenu (pourquoi lui : notebooks/1_comparaison_modeles.ipynb) --> calibration sigmoïde et seuil
choisis sur la validation --> métriques sur le test --> MLflow. Le @champion n'est remplacé que si le gain net en
validation ne baisse pas. Usage : python -m src.train
"""

import logging
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import ALIAS_CHAMPION, CIBLE, NOM_EXPERIENCE, NOM_MODELE, URI_MLFLOW
from src.donnees import DATE_TEST, DATE_VALIDATION, HORIZON, MACHINES_RESERVEES, charger, decouper, preparer_X
from src.evaluation import (
    BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR,
    PAS_HORIZON,
    bilan_economique,
    grille_seuils,
    metriques_evenementielles,
    trouver_seuil_pour_budget_fa,
)
from src.metier import COUT_INSPECTION, DUREE_INTERVENTION_PLANIFIEE_H

logger = logging.getLogger(__name__)

# Peu de feuilles et forte régularisation : le signal de panne est rare
HYPERPARAMETRES_LIGHTGBM: dict[str, Any] = {"n_estimators": 150, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 80,
                            "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.3, "reg_lambda": 0.3}  # fmt: skip
METRIQUE_PROMOTION = "gain_net_validation"  # décidée sur la validation : le test ne sert qu'au rapport


class LGBMLogOdds(LGBMClassifier):
    """LightGBM qui expose ses log-odds : la calibration sigmoïde s'ajuste sur un score non borné, donc sans plancher."""

    def decision_function(self, X):
        return self.predict(X, raw_score=True)


def nouveau_modele() -> LGBMLogOdds:
    return LGBMLogOdds(**HYPERPARAMETRES_LIGHTGBM, random_state=42, n_jobs=-1, verbose=-1)


def entrainer(df: pd.DataFrame) -> tuple[CalibratedClassifierCV, float, dict]:
    train, validation, test = decouper(df)
    logger.info(f"Train {len(train):,} | validation {len(validation):,} | test {len(test):,} lignes")
    # Le modèle n'apprend et ne note que les machines en marche : à l'arrêt, les capteurs tombent à zéro
    marche_train, marche_validation = train[train["etat"] == "marche"], validation[validation["etat"] == "marche"]
    modele = nouveau_modele().fit(preparer_X(marche_train), marche_train[CIBLE])
    calibre = CalibratedClassifierCV(FrozenEstimator(modele), method="sigmoid").fit(
        preparer_X(marche_validation), marche_validation[CIBLE]
    )
    probas_validation = calibre.predict_proba(preparer_X(validation))[:, 1]
    seuil, evenements_validation = trouver_seuil_pour_budget_fa(
        validation, probas_validation, grille_seuils(probas_validation), PAS_HORIZON, BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR
    )

    probas = calibre.predict_proba(preparer_X(test))[:, 1]
    evenements = metriques_evenementielles(test, probas, seuil, PAS_HORIZON)
    bilan = bilan_economique(test, evenements)
    reservees = test["id_machine"].isin(MACHINES_RESERVEES).to_numpy()
    en_marche = (test["etat"] == "marche").to_numpy()
    cible, probas_marche = test.loc[en_marche, CIBLE], probas[en_marche]
    metriques = {
        "gain_net_validation": bilan_economique(validation, evenements_validation)["gain_net"],
        **{k: evenements[k] for k in ("recall_evenementiel", "recall_preavis_4h", "precision_alertes")},
        "taux_fausses_alertes_par_machine_jour": evenements["taux_fausses_alertes_par_machine_jour"],
        "gain_net_fcfa": bilan["gain_net"],
        "cout_inspection_rentable_fcfa": min(bilan["cout_inspection_rentable"], 1e9),  # infini sans fausse alerte
        "n_failures_test": evenements["nb_pannes"],
        "n_caught_test": evenements["nb_attrapees"],
        "preavis_median_h": float(np.median(evenements["preavis_heures"])) if evenements["preavis_heures"] else 0.0,
        "recall_machines_reservees": metriques_evenementielles(test[reservees], probas[reservees], seuil, PAS_HORIZON)[
            "recall_evenementiel"
        ],
        "pr_auc": average_precision_score(cible, probas_marche),
        "roc_auc": roc_auc_score(cible, probas_marche),
        "brier_score_raw": brier_score_loss(cible, modele.predict_proba(preparer_X(test[en_marche]))[:, 1]),
        "brier_score_calibrated": brier_score_loss(cible, probas_marche),
    }
    return calibre, seuil, metriques


def promouvoir_si_meilleur(client: mlflow.MlflowClient, version: str, score: float) -> bool:
    """@champion si le gain net en validation égale ou dépasse celui du champion (automatique s'il n'y en a pas)."""
    try:
        champion = client.get_model_version_by_alias(NOM_MODELE, ALIAS_CHAMPION)
        score_champion = client.get_run(str(champion.run_id)).data.metrics.get(METRIQUE_PROMOTION, float("-inf"))
    except Exception:
        score_champion = float("-inf")
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
    with mlflow.start_run(run_name="lightgbm_entrainement"):
        mlflow.log_params({
            "modele": "lightgbm", **HYPERPARAMETRES_LIGHTGBM,
            "seuil_optimal": seuil,  # valeur exacte : relue par l'API avec le modèle
            "budget_fa_machine_jour": BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR, "cout_inspection_fcfa": COUT_INSPECTION,
            "duree_intervention_planifiee_h": DUREE_INTERVENTION_PLANIFIEE_H, "horizon_heures": HORIZON / pd.Timedelta("1h"),
            "date_validation": str(DATE_VALIDATION.date()), "date_test": str(DATE_TEST.date()),
        })  # fmt: skip
        mlflow.log_metrics(metriques)
        info = mlflow.sklearn.log_model(sk_model=calibre, name="model_pipeline", registered_model_name=NOM_MODELE)
    promouvoir_si_meilleur(mlflow.MlflowClient(), info.registered_model_version, metriques[METRIQUE_PROMOTION])
    for nom, valeur in metriques.items():
        print(f"{nom}: {valeur:.4f}" if isinstance(valeur, float) else f"{nom}: {valeur}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
