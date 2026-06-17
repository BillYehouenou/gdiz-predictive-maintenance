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
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from src.configloader import load_config
from src.preprocessor import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SPLIT_DATE = "2025-01-01"
THRESHOLD_PATH = "models/optimal_threshold.json"


class ModelPipeline:
    def __init__(self):
        self.config = load_config()
        self.target_col = self.config["dataset"]["target_column"]
        self.features_cols = self.config["dataset"]["features"]
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.config["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)

    def _temporal_split(self, df: pd.DataFrame):
        """Split temporel : train sur 2024, test sur 2025. Évite le data leakage."""
        if "timestamp" not in df.columns:
            raise ValueError("La colonne 'timestamp' est requise pour le split temporel.")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        train = df[df["timestamp"] < SPLIT_DATE]
        test = df[df["timestamp"] >= SPLIT_DATE]
        logger.info(f"Split temporel - Train: {len(train):,} lignes (< {SPLIT_DATE}) | Test: {len(test):,} lignes")
        return (
            train[self.features_cols], train[self.target_col],
            test[self.features_cols], test[self.target_col],
        )

    def _find_optimal_threshold(self, probas: np.ndarray, y_true: np.ndarray) -> float:
        """Seuil optimal sur F2-score (2 fois plus de poids au Recall qu'à la Précision)."""
        thresholds = np.arange(0.005, 0.50, 0.005)
        f2_scores = [
            fbeta_score(y_true, (probas >= t).astype(int), beta=2, zero_division=0)
            for t in thresholds
        ]
        best_idx = int(np.argmax(f2_scores))
        best_t = float(thresholds[best_idx])
        logger.info(f"Seuil optimal F2 : {best_t:.3f} (F2={f2_scores[best_idx]:.4f})")
        return best_t

    def _log_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        ax.set_title("Confusion Matrix (seuil F2 optimal)")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, bbox_inches="tight", dpi=100)
            mlflow.log_artifact(tmp.name, artifact_path="plots")
        plt.close(fig)

    def _log_feature_importances(self, full_pipeline: Pipeline):
        try:
            lgbm = full_pipeline.named_steps["classifier"]
            importances = lgbm.feature_importances_.tolist()
            fi_dict = {str(i): float(v) for i, v in enumerate(importances)}
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as tmp:
                json.dump(fi_dict, tmp)
                mlflow.log_artifact(tmp.name, artifact_path="feature_importances")
        except AttributeError:
            logger.warning("feature_importances_ non disponible pour ce modèle.")

    def run_training(self, model, train_df: pd.DataFrame, model_name: str, hyperparameters: dict):
        """Pipeline d'entraînement complet : split temporel - fit - métriques - seuil F2 - MLflow."""
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        logger.info("Début du pipeline d'entraînement...")

        X_train, y_train, X_test, y_test = self._temporal_split(train_df)
        logger.info(f"Pannes train: {y_train.sum():,} / {len(y_train):,} ({y_train.mean()*100:.2f}%)")
        logger.info(f"Pannes test:  {y_test.sum():,} / {len(y_test):,} ({y_test.mean()*100:.2f}%)")

        with mlflow.start_run(run_name=f"{model_name}_Training"):
            preprocessor = Preprocessor()
            full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

            logger.info("Entraînement du modèle...")
            full_pipeline.fit(X_train, y_train)
            logger.info("Entraînement terminé.")

            probas_test = full_pipeline.predict_proba(X_test)[:, 1]

            optimal_threshold = self._find_optimal_threshold(probas_test, y_test)

            y_pred = (probas_test >= optimal_threshold).astype(int)

            metrics = {
                "roc_auc": roc_auc_score(y_test, probas_test),
                "precision": precision_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0),
                "recall": recall_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0),
                "f2_score": fbeta_score(y_test, y_pred, beta=2, pos_label=1, zero_division=0),
                "failure_rate_train": float(y_train.mean()),
                "failure_rate_test": float(y_test.mean()),
            }

            mlflow.log_params(hyperparameters)
            mlflow.log_param("dataset_name", self.config["dataset"]["name"])
            mlflow.log_param("split_date", SPLIT_DATE)
            mlflow.log_param("optimal_threshold", round(optimal_threshold, 4))
            mlflow.log_param("n_train", len(y_train))
            mlflow.log_param("n_test", len(y_test))
            mlflow.log_metrics(metrics)

            self._log_confusion_matrix(y_test.values, y_pred)
            self._log_feature_importances(full_pipeline)

            mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                name="model_pipeline",
                registered_model_name=model_name,
            )

            os.makedirs("models", exist_ok=True)
            joblib.dump(full_pipeline, "models/model_pipeline.pkl")
            with open(THRESHOLD_PATH, "w") as f:
                json.dump({"threshold": optimal_threshold}, f)

            logger.info("Pipeline sauvegardé : models/model_pipeline.pkl")
            logger.info(f"Seuil optimal sauvegardé : {THRESHOLD_PATH}")

            for name, val in metrics.items():
                print(f"{name}: {val:.4f}")
