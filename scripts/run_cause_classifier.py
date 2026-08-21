"""
Étage 2 — diagnostic de cause (point 14) — uv run python scripts/run_cause_classifier.py

Ne remplace pas l'étage 1 (l'alerte "faut-il s'inquiéter", mutualisée et déjà
optimale — cf. docs/model-decisions.md). S'exécute seulement sur les lignes déjà
signalées par l'étage 1, pour répondre à "laquelle des 3 causes est la plus
probable" — un diagnostic, pas un détecteur.
"""

import logging

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from src.configloader import charger_config
from src.dataloader import ChargeurDonnees
from src.features import MODES_PANNE_PREDICTIBLES, calculer_label_cause_panne
from src.preprocessor import Preprocesseur
from src.train import DATE_COUPURE, DATE_COUPURE_VALIDATION

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = charger_config()
    colonnes_features = config["jeu_donnees"]["features"]

    df = ChargeurDonnees().charger_donnees_brutes()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["label_cause"] = calculer_label_cause_panne(df, heures_horizon=120, modes_panne=MODES_PANNE_PREDICTIBLES)

    # Entraîné et évalué UNIQUEMENT sur la population pré-panne (là où l'étage 1
    # alerterait déjà) — pas sur les machines saines, qui n'ont pas de cause à prédire.
    pre_panne = df[df["label_cause"].notna()].copy()

    train = pre_panne[pre_panne["timestamp"] < DATE_COUPURE_VALIDATION]
    validation = pre_panne[(pre_panne["timestamp"] >= DATE_COUPURE_VALIDATION) & (pre_panne["timestamp"] < DATE_COUPURE)]
    test = pre_panne[pre_panne["timestamp"] >= DATE_COUPURE]
    logger.info(f"Lignes pré-panne — train: {len(train):,} | validation: {len(validation):,} | test: {len(test):,}")
    logger.info(f"Répartition des causes (train): {train['label_cause'].value_counts().to_dict()}")

    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=30,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", Preprocesseur()), ("classifier", model)])
    pipeline.fit(train[colonnes_features], train["label_cause"])

    for nom, sous_ensemble in [("validation", validation), ("test", test)]:
        y_true = sous_ensemble["label_cause"]
        y_predit = pipeline.predict(sous_ensemble[colonnes_features])
        acc = accuracy_score(y_true, y_predit)
        logger.info(f"--- {nom} : accuracy={acc:.1%} ---")
        print(classification_report(y_true, y_predit, zero_division=0))


if __name__ == "__main__":
    main()
