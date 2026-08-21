"""
Baselines de référence — uv run python scripts/run_baseline_comparison.py

Point 23 : le modèle LightGBM doit battre une règle triviale ("usure > seuil") et
une régression logistique, sur la métrique événementielle (recall @ budget de
fausses alertes) — pas juste exister. Résultats dans docs/experiments.md.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.configloader import load_config
from src.dataloader import DataLoader
from src.evaluation import find_threshold_for_fa_budget
from src.preprocessor import Preprocessor
from src.train import (
    EVENT_HORIZON_STEPS,
    FALSE_ALARM_BUDGET_PER_MACHINE_DAY,
    SPLIT_DATE,
    VALIDATION_SPLIT_DATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RULE_THRESHOLD_GRID = np.arange(5.0, 100.0, 1.0)  # sur l'échelle brute de tool_wear (%), pas une probabilité
LR_THRESHOLD_GRID = np.arange(0.005, 0.95, 0.005)


def _split(df: pd.DataFrame, features_cols: list[str], target_col: str):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    train = df[df["timestamp"] < VALIDATION_SPLIT_DATE]
    val = df[(df["timestamp"] >= VALIDATION_SPLIT_DATE) & (df["timestamp"] < SPLIT_DATE)]
    test = df[df["timestamp"] >= SPLIT_DATE]
    meta_cols = ["machine_id", "timestamp", "target", "failure_mode"]
    return (
        train[features_cols],
        train[target_col],
        val[features_cols],
        val[target_col],
        val[meta_cols],
        test[features_cols],
        test[target_col],
        test[meta_cols],
    )


def evaluate_baseline(name: str, probas_val, probas_test, val_meta, test_meta, threshold_grid):
    threshold, val_metrics = find_threshold_for_fa_budget(
        val_meta, probas_val, threshold_grid, EVENT_HORIZON_STEPS, FALSE_ALARM_BUDGET_PER_MACHINE_DAY
    )
    from src.evaluation import event_based_metrics

    test_metrics = event_based_metrics(test_meta, probas_test, threshold, EVENT_HORIZON_STEPS)
    logger.info(
        f"{name} — seuil={threshold:.3f} | validation recall_event={val_metrics['recall_event']:.3f} | "
        f"TEST recall_event={test_metrics['recall_event']:.3f} FA/machine/jour="
        f"{test_metrics['false_alarm_rate_per_machine_day']:.3f} "
        f"({test_metrics['n_caught']}/{test_metrics['n_failures']} pannes attrapées)"
    )
    return test_metrics


def main():
    config = load_config()
    features_cols = config["dataset"]["features"]
    target_col = config["dataset"]["target_column"]

    df = DataLoader().load_raw_data()
    X_train, y_train, X_val, y_val, val_meta, X_test, y_test, test_meta = _split(df, features_cols, target_col)

    # 1. Règle triviale : usure > seuil. Pas d'apprentissage, juste un balayage de
    # seuil sur tool_wear directement (même infrastructure d'évaluation que le modèle).
    probas_val_rule = X_val["tool_wear"].to_numpy()
    probas_test_rule = X_test["tool_wear"].to_numpy()
    evaluate_baseline("Règle (tool_wear > seuil)", probas_val_rule, probas_test_rule, val_meta, test_meta, RULE_THRESHOLD_GRID)

    # 2. Régression logistique : même preprocessing que le modèle principal, pas
    # d'ingénierie spécifique — le point de comparaison le plus juste.
    lr_pipeline = Pipeline(
        steps=[("preprocessor", Preprocessor()), ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))]
    )
    lr_pipeline.fit(X_train, y_train)
    probas_val_lr = lr_pipeline.predict_proba(X_val)[:, 1]
    probas_test_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    evaluate_baseline("Régression logistique", probas_val_lr, probas_test_lr, val_meta, test_meta, LR_THRESHOLD_GRID)

    logger.info("Comparer ces résultats à ceux du modèle LightGBM (`make train`) — chiffres à jour dans docs/experiments.md.")


if __name__ == "__main__":
    main()
