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

from src.configloader import load_config
from src.dataloader import DataLoader
from src.features import PREDICTABLE_FAILURE_MODES, compute_failure_cause_label
from src.preprocessor import Preprocessor
from src.train import SPLIT_DATE, VALIDATION_SPLIT_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    features_cols = config["dataset"]["features"]

    df = DataLoader().load_raw_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cause_label"] = compute_failure_cause_label(df, horizon_hours=120, failure_modes=PREDICTABLE_FAILURE_MODES)

    # Entraîné et évalué UNIQUEMENT sur la population pré-panne (là où l'étage 1
    # alerterait déjà) — pas sur les machines saines, qui n'ont pas de cause à prédire.
    pre_failure = df[df["cause_label"].notna()].copy()

    train = pre_failure[pre_failure["timestamp"] < VALIDATION_SPLIT_DATE]
    val = pre_failure[(pre_failure["timestamp"] >= VALIDATION_SPLIT_DATE) & (pre_failure["timestamp"] < SPLIT_DATE)]
    test = pre_failure[pre_failure["timestamp"] >= SPLIT_DATE]
    logger.info(f"Lignes pré-panne — train: {len(train):,} | validation: {len(val):,} | test: {len(test):,}")
    logger.info(f"Répartition des causes (train): {train['cause_label'].value_counts().to_dict()}")

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
    pipeline = Pipeline(steps=[("preprocessor", Preprocessor()), ("classifier", model)])
    pipeline.fit(train[features_cols], train["cause_label"])

    for name, split in [("validation", val), ("test", test)]:
        y_true = split["cause_label"]
        y_pred = pipeline.predict(split[features_cols])
        acc = accuracy_score(y_true, y_pred)
        logger.info(f"--- {name} : accuracy={acc:.1%} ---")
        print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
