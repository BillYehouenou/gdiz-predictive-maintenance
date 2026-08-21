import json
import logging
import os
import tempfile

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.configloader import load_config
from src.evaluation import event_based_metrics, find_threshold_for_fa_budget
from src.preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Validation : sert uniquement à choisir le seuil, jamais à entraîner ni à rapporter
# les métriques finales — le split à 3 voies évite le biais optimiste d'un seuil
# ajusté sur les données mêmes qui servent à le rapporter (point 16 de la roadmap).
VALIDATION_SPLIT_DATE = "2024-10-01"
SPLIT_DATE = "2025-01-01"
THRESHOLD_PATH = "models/optimal_threshold.json"
CHAMPION_ALIAS = "champion"
PROMOTION_METRIC = "recall_event"

# Horizon événementiel : aligné sur target_h120_predictable (cf. config/config.yaml),
# l'horizon au-delà duquel le modèle n'a de toute façon aucun signal d'entraînement.
EVENT_HORIZON_HOURS = 120
EVENT_HORIZON_STEPS = EVENT_HORIZON_HOURS * 2  # pas de 30 min

# Budget de fausses alertes : combien de dérangements par machine et par jour
# l'équipe de maintenance accepte pour une alerte qui ne mène à rien. Remplace le
# F2 comme objectif de sélection du seuil (point 17/24) — cf. docs/experiments.md
# pour la courbe recall/FA mesurée qui justifie cette valeur.
FALSE_ALARM_BUDGET_PER_MACHINE_DAY = 0.2
THRESHOLD_GRID = np.arange(0.005, 0.50, 0.005)


class ModelPipeline:
    def __init__(self):
        self.config = load_config()
        self.target_col = self.config["dataset"]["target_column"]
        self.features_cols = self.config["dataset"]["features"]
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.config["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.MlflowClient()

    def _temporal_split(self, df: pd.DataFrame):
        """
        Split temporel à 3 voies : train (< VALIDATION_SPLIT_DATE), validation
        ([VALIDATION_SPLIT_DATE, SPLIT_DATE) — sert uniquement à choisir le seuil),
        test (>= SPLIT_DATE — jamais vu avant le rapport final). Le seuil est choisi
        sur la validation, pas sur le test, pour éviter le biais optimiste d'un seuil
        ajusté sur les données mêmes qui servent à le rapporter.
        """
        if "timestamp" not in df.columns:
            raise ValueError("La colonne 'timestamp' est requise pour le split temporel.")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        train = df[df["timestamp"] < VALIDATION_SPLIT_DATE]
        validation = df[(df["timestamp"] >= VALIDATION_SPLIT_DATE) & (df["timestamp"] < SPLIT_DATE)]
        test = df[df["timestamp"] >= SPLIT_DATE]
        logger.info(
            f"Split temporel - Train: {len(train):,} (< {VALIDATION_SPLIT_DATE}) | "
            f"Validation: {len(validation):,} ([{VALIDATION_SPLIT_DATE}, {SPLIT_DATE})) | "
            f"Test: {len(test):,} (>= {SPLIT_DATE})"
        )
        return (
            train[self.features_cols],
            train[self.target_col],
            validation[self.features_cols],
            validation[self.target_col],
            test[self.features_cols],
            test[self.target_col],
        )

    def _log_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        ax.set_title("Confusion Matrix (seuil recall événementiel @ budget FA)")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, bbox_inches="tight", dpi=100)
            mlflow.log_artifact(tmp.name, artifact_path="plots")
        plt.close(fig)

    def _log_feature_importances(self, full_pipeline: Pipeline):
        try:
            lgbm = full_pipeline.named_steps["classifier"]
            importances = lgbm.feature_importances_.tolist()
            fi_dict = {str(i): float(v) for i, v in enumerate(importances)}
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
                json.dump(fi_dict, tmp)
                mlflow.log_artifact(tmp.name, artifact_path="feature_importances")
        except AttributeError:
            logger.warning("feature_importances_ non disponible pour ce modèle.")

    def _promote_if_better(self, model_name: str, new_version: str, new_score: float) -> bool:
        """
        Gate de promotion : la nouvelle version ne devient '@champion' que si elle égale
        ou dépasse le champion actuel sur PROMOTION_METRIC. Sans champion existant
        (premier entraînement), la promotion est automatique. Évite qu'un ré-entraînement
        dégradé ne remplace silencieusement le modèle servi en production.
        """
        try:
            champion = self.client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
            champion_score = self.client.get_run(champion.run_id).data.metrics.get(PROMOTION_METRIC, -1.0)
        except Exception:
            champion, champion_score = None, -1.0

        if new_score >= champion_score:
            self.client.set_registered_model_alias(model_name, CHAMPION_ALIAS, new_version)
            logger.info(
                f"Version {new_version} promue @{CHAMPION_ALIAS} ({PROMOTION_METRIC}={new_score:.4f} >= {champion_score:.4f})"
            )
            return True

        logger.warning(
            f"Version {new_version} ({PROMOTION_METRIC}={new_score:.4f}) n'a pas dépassé le champion actuel "
            f"(version {champion.version}, {PROMOTION_METRIC}={champion_score:.4f}) — champion conservé."
        )
        return False

    def run_training(self, model, train_df: pd.DataFrame, model_name: str, hyperparameters: dict):
        """
        Pipeline d'entraînement complet : split temporel à 3 voies - fit - seuil choisi
        sur validation (recall événementiel sous budget de fausses alertes) - métriques
        rapportées sur test - MLflow.
        """
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        logger.info("Début du pipeline d'entraînement...")

        X_train, y_train, X_val, y_val, X_test, y_test = self._temporal_split(train_df)
        logger.info(f"Pannes train:      {y_train.sum():,} / {len(y_train):,} ({y_train.mean() * 100:.2f}%)")
        logger.info(f"Pannes validation: {y_val.sum():,} / {len(y_val):,} ({y_val.mean() * 100:.2f}%)")
        logger.info(f"Pannes test:       {y_test.sum():,} / {len(y_test):,} ({y_test.mean() * 100:.2f}%)")

        with mlflow.start_run(run_name=f"{model_name}_Training"):
            preprocessor = Preprocessor()
            full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

            logger.info("Entraînement du modèle...")
            full_pipeline.fit(X_train, y_train)
            logger.info("Entraînement terminé.")

            probas_val = full_pipeline.predict_proba(X_val)[:, 1]
            probas_test = full_pipeline.predict_proba(X_test)[:, 1]

            # Seuil choisi sur la validation, jamais sur le test (point 16).
            val_meta = train_df.loc[X_val.index, ["machine_id", "timestamp", "target", "failure_mode"]]
            optimal_threshold, val_event_metrics = find_threshold_for_fa_budget(
                val_meta, probas_val, THRESHOLD_GRID, EVENT_HORIZON_STEPS, FALSE_ALARM_BUDGET_PER_MACHINE_DAY
            )
            logger.info(
                f"Seuil retenu (validation) : {optimal_threshold:.3f} — "
                f"recall_event={val_event_metrics['recall_event']:.3f}, "
                f"FA/machine/jour={val_event_metrics['false_alarm_rate_per_machine_day']:.3f}"
            )

            # Métriques rapportées sur le test, jamais utilisé pour choisir le seuil.
            test_meta = train_df.loc[X_test.index, ["machine_id", "timestamp", "target", "failure_mode"]]
            test_event_metrics = event_based_metrics(test_meta, probas_test, optimal_threshold, EVENT_HORIZON_STEPS)
            y_pred = (probas_test >= optimal_threshold).astype(int)

            metrics = {
                "recall_event": test_event_metrics["recall_event"],
                "false_alarm_rate_per_machine_day": test_event_metrics["false_alarm_rate_per_machine_day"],
                "n_failures_test": test_event_metrics["n_failures"],
                "n_caught_test": test_event_metrics["n_caught"],
                "n_false_alarm_episodes_test": test_event_metrics["n_false_alarm_episodes"],
                "pr_auc": average_precision_score(y_test, probas_test),
                "roc_auc": roc_auc_score(y_test, probas_test),
                "precision": precision_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0),
                "recall": recall_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0),
                "failure_rate_train": float(y_train.mean()),
                "failure_rate_validation": float(y_val.mean()),
                "failure_rate_test": float(y_test.mean()),
            }

            mlflow.log_params(hyperparameters)
            mlflow.log_param("dataset_name", self.config["dataset"]["name"])
            mlflow.log_param("validation_split_date", VALIDATION_SPLIT_DATE)
            mlflow.log_param("split_date", SPLIT_DATE)
            mlflow.log_param("optimal_threshold", round(optimal_threshold, 4))
            mlflow.log_param("false_alarm_budget_per_machine_day", FALSE_ALARM_BUDGET_PER_MACHINE_DAY)
            mlflow.log_param("event_horizon_hours", EVENT_HORIZON_HOURS)
            mlflow.log_param("n_train", len(y_train))
            mlflow.log_param("n_validation", len(y_val))
            mlflow.log_param("n_test", len(y_test))
            mlflow.log_metrics(metrics)

            self._log_confusion_matrix(y_test.values, y_pred)
            self._log_feature_importances(full_pipeline)

            model_info = mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                name="model_pipeline",
                registered_model_name=model_name,
            )

            promoted = self._promote_if_better(model_name, model_info.registered_model_version, metrics[PROMOTION_METRIC])

            if promoted:
                os.makedirs("models", exist_ok=True)
                joblib.dump(full_pipeline, "models/model_pipeline.pkl")
                with open(THRESHOLD_PATH, "w") as f:
                    json.dump({"threshold": optimal_threshold}, f)
                logger.info("Pipeline sauvegardé : models/model_pipeline.pkl")
                logger.info(f"Seuil optimal sauvegardé : {THRESHOLD_PATH}")
            else:
                logger.info("Fallback local non mis à jour (version non promue).")

            for name, val in metrics.items():
                print(f"{name}: {val}")
