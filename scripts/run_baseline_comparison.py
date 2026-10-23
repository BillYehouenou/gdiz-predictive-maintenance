"""
Baselines de référence

Le modèle LightGBM doit battre une règle triviale et une régression logistique,
sur la métrique événementielle (recall @ budget de fausses alertes).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.configloader import charger_config
from src.dataloader import ChargeurDonnees
from src.evaluation import trouver_seuil_pour_budget_fa
from src.preprocessor import Preprocesseur
from src.train import (
    BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR,
    DATE_COUPURE,
    DATE_COUPURE_VALIDATION,
    PAS_HORIZON_EVENEMENT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLONNE_REGLE = "heures_depuis_changement_outil"
GRILLE_SEUILS_RL = np.arange(0.005, 0.95, 0.005)


def _split(df: pd.DataFrame, colonnes_features: list[str], colonne_cible: str):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    train = df[df["timestamp"] < DATE_COUPURE_VALIDATION]
    val = df[(df["timestamp"] >= DATE_COUPURE_VALIDATION) & (df["timestamp"] < DATE_COUPURE)]
    test = df[df["timestamp"] >= DATE_COUPURE]
    colonnes_meta = ["id_machine", "timestamp", "panne", "mode_panne"]
    return (
        train[colonnes_features],
        train[colonne_cible],
        val[colonnes_features],
        val[colonne_cible],
        val[colonnes_meta],
        test[colonnes_features],
        test[colonne_cible],
        test[colonnes_meta],
    )


def evaluer_baseline(nom: str, probas_validation, probas_test, meta_validation, meta_test, grille_seuils):
    seuil, metriques_validation = trouver_seuil_pour_budget_fa(
        meta_validation, probas_validation, grille_seuils, PAS_HORIZON_EVENEMENT, BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR
    )
    from src.evaluation import metriques_evenementielles

    metriques_test = metriques_evenementielles(meta_test, probas_test, seuil, PAS_HORIZON_EVENEMENT)
    logger.info(
        f"{nom} — seuil={seuil:.3f} | validation recall_evenementiel={metriques_validation['recall_evenementiel']:.3f} | "
        f"TEST recall_evenementiel={metriques_test['recall_evenementiel']:.3f} FA/machine/jour="
        f"{metriques_test['taux_fausses_alertes_par_machine_jour']:.3f} "
        f"({metriques_test['nb_attrapees']}/{metriques_test['nb_pannes']} pannes attrapées)"
    )
    return metriques_test


def main():
    config = charger_config()
    colonnes_features = config["jeu_donnees"]["features"]
    colonne_cible = config["jeu_donnees"]["colonne_cible"]

    df = ChargeurDonnees().charger_donnees_brutes()
    X_train, y_train, X_val, y_val, meta_validation, X_test, y_test, meta_test = _split(df, colonnes_features, colonne_cible)

    # 1. Règle triviale : heures d'outil > seuil. Pas d'apprentissage, juste un balayage de
    # seuil sur le compteur GMAO (même infrastructure d'évaluation que le modèle).
    probas_val_regle = X_val[COLONNE_REGLE].to_numpy()
    probas_test_regle = X_test[COLONNE_REGLE].to_numpy()
    grille_regle = np.unique(np.quantile(probas_val_regle, np.linspace(0.5, 0.999, 100)))  # échelle brute, pas une probabilité
    evaluer_baseline("Règle triviale", probas_val_regle, probas_test_regle, meta_validation, meta_test, grille_regle)

    # 2. Régression logistique : même preprocessing que le modèle principal, pas
    # d'ingénierie spécifique — le point de comparaison le plus juste.
    pipeline_rl = Pipeline(
        steps=[("preprocessor", Preprocesseur()), ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))]
    )
    pipeline_rl.fit(X_train, y_train)
    probas_val_rl = pipeline_rl.predict_proba(X_val)[:, 1]
    probas_test_rl = pipeline_rl.predict_proba(X_test)[:, 1]
    evaluer_baseline("Régression logistique", probas_val_rl, probas_test_rl, meta_validation, meta_test, GRILLE_SEUILS_RL)

    logger.info("Comparer ces résultats à ceux du modèle LightGBM (`make train`) — chiffres à jour dans docs/experiments.md.")


if __name__ == "__main__":
    main()
