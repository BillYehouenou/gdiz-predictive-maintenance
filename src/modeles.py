"""
Modèles candidats, du plus simple au plus complexe, comparés sur trois fenêtres de validation (T2, T3, T4 2024)
puis sur le test 2025. Usage : python -m src.modeles

Résultat (octobre 2026) : le boosting bat nettement la règle, la régression logistique et l'arbre. Entre variantes de
boosting, les écarts restent dans l'incertitude (~20 pannes par fenêtre) : LightGBM 150 arbres est conservé.
"""

import logging
import pickle
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import CIBLE, URI_MLFLOW
from src.donnees import PLIS, charger, decouper, decouper_pli
from src.evaluation import (
    BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR,
    PAS_HORIZON,
    grille_seuils,
    metriques_evenementielles,
    trouver_seuil_pour_budget_fa,
)
from src.features import FEATURES, preparer_X

logger = logging.getLogger(__name__)

MODELE_RETENU = "lightgbm"
# Peu de feuilles et forte régularisation : le signal de panne est rare
HYPERPARAMETRES_LIGHTGBM: dict[str, Any] = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 80,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.3,
    "reg_lambda": 0.3,
}
NUMERIQUES = [f for f in FEATURES if f != "type_machine"]


class RegleSeuil(ClassifierMixin, BaseEstimator):
    """Score = une seule colonne (par défaut les heures d'outil). Aucun apprentissage."""

    def __init__(self, colonne: str = "heures_depuis_changement_outil"):
        self.colonne = colonne

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        score = X[self.colonne].fillna(0).to_numpy(dtype=float)
        return np.column_stack([-score, score])


class LGBMLogOdds(LGBMClassifier):
    """
    LightGBM qui expose ses log-odds. La calibration sigmoïde s'ajuste alors sur un score non borné : sur la probabilité
    brute (bornée à 0), elle imposait un plancher d'environ 2,8 % à toutes les machines saines.
    """

    def decision_function(self, X):
        return self.predict(X, raw_score=True)


def _encodeur(numerique) -> object:
    return make_column_transformer((numerique, NUMERIQUES), (OneHotEncoder(handle_unknown="ignore"), ["type_machine"]))


# Ordre = complexité croissante
CANDIDATS = {
    "regle_heures_outil": lambda: RegleSeuil(),
    "regression_logistique": lambda: make_pipeline(
        _encodeur(make_pipeline(SimpleImputer(strategy="median"), StandardScaler())),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    ),
    "arbre_decision": lambda: make_pipeline(
        _encodeur("passthrough"), DecisionTreeClassifier(max_depth=6, min_samples_leaf=500, random_state=42)
    ),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        categorical_features="from_dtype",
        random_state=42,
    ),
    "lightgbm": lambda: LGBMLogOdds(**HYPERPARAMETRES_LIGHTGBM, random_state=42, n_jobs=-1, verbose=-1),
}


def recall_au_budget(validation: pd.DataFrame, scores: np.ndarray) -> tuple[float, dict]:
    """Seuil fixé sur la fenêtre elle-même : mesure la capacité à classer, au point de fonctionnement de l'atelier."""
    return trouver_seuil_pour_budget_fa(
        validation, scores, grille_seuils(scores), PAS_HORIZON, BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR
    )


def evaluer(nom: str, df: pd.DataFrame, test: pd.DataFrame) -> dict:
    resultat: dict = {"modele": nom}
    for i, (debut, fin) in enumerate(PLIS, start=1):
        train, validation = decouper_pli(df, debut, fin)
        depart = time.perf_counter()
        modele = CANDIDATS[nom]().fit(preparer_X(train), train[CIBLE])
        resultat["entrainement_s"] = time.perf_counter() - depart
        seuil, m = recall_au_budget(validation, modele.predict_proba(preparer_X(validation))[:, 1])
        resultat[f"recall_pli_{i}"] = m["recall_evenementiel"]
        resultat[f"pannes_pli_{i}"] = m["nb_pannes"]

    # Dernier pli = protocole de déploiement : son seuil, appliqué tel quel au test 2025
    X_test = preparer_X(test)
    depart = time.perf_counter()
    scores_test = modele.predict_proba(X_test)[:, 1]
    resultat["latence_ms_1000_lignes"] = (time.perf_counter() - depart) / len(test) * 1e6
    m = metriques_evenementielles(test, scores_test, seuil, PAS_HORIZON)
    resultat["recall_test"] = m["recall_evenementiel"]
    resultat["fa_test"] = m["taux_fausses_alertes_par_machine_jour"]
    resultat["taille_ko"] = len(pickle.dumps(modele)) / 1024
    resultat["recall_validation"] = float(np.mean([resultat[f"recall_pli_{i}"] for i in range(1, len(PLIS) + 1)]))
    # Écart-type binomial : sous ~2 écarts-types, deux modèles ne sont pas départagés
    pannes_validation = sum(resultat[f"pannes_pli_{i}"] for i in range(1, len(PLIS) + 1))
    resultat["ecart_type_validation"] = _ecart_type(resultat["recall_validation"], pannes_validation)
    resultat["ecart_type_test"] = _ecart_type(m["recall_evenementiel"], m["nb_pannes"])
    return resultat


def _ecart_type(recall: float, nb_pannes: int) -> float:
    return float(np.sqrt(recall * (1 - recall) / nb_pannes)) if nb_pannes else 0.0


def main():
    df = charger()
    _, _, test = decouper(df)
    resultats = pd.DataFrame([evaluer(nom, df, test) for nom in CANDIDATS]).set_index("modele")

    mlflow.set_tracking_uri(URI_MLFLOW)
    mlflow.set_experiment("GDIZ_Comparaison_Modeles")
    for nom, ligne in resultats.iterrows():
        with mlflow.start_run(run_name=str(nom)):
            mlflow.log_metrics(ligne.to_dict())
            mlflow.set_tag("retenu", str(nom == MODELE_RETENU))

    colonnes = ["recall_pli_1", "recall_pli_2", "recall_pli_3", "recall_validation", "ecart_type_validation"]
    colonnes += ["recall_test", "ecart_type_test", "fa_test"]
    colonnes += ["entrainement_s", "latence_ms_1000_lignes", "taille_ko"]
    pannes = ", ".join(str(int(resultats.iloc[0][f"pannes_pli_{i}"])) for i in range(1, len(PLIS) + 1))
    print(f"\nPannes par pli de validation : {pannes}\n")
    print(resultats[colonnes].round(3).to_string())
    print(f"\nModèle déployé : {MODELE_RETENU}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
