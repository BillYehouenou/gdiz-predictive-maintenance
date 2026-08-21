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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.configloader import charger_config
from src.evaluation import metriques_evenementielles, trouver_seuil_pour_budget_fa
from src.preprocessor import Preprocesseur

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Validation : sert uniquement à choisir le seuil, jamais à entraîner ni à rapporter
# les métriques finales — le split à 3 voies évite le biais optimiste d'un seuil
# ajusté sur les données mêmes qui servent à le rapporter (point 16 de la roadmap).
DATE_COUPURE_VALIDATION = "2024-10-01"
DATE_COUPURE = "2025-01-01"
CHEMIN_SEUIL = "models/seuil_optimal.json"
ALIAS_CHAMPION = "champion"
METRIQUE_PROMOTION = "recall_evenementiel"

# Horizon événementiel : aligné sur cible_h120_predictible (cf. config/config.yaml).
# Des horizons différenciés par mode (point 15, HORIZON_PAR_MODE_PANNE dans
# src/features.py) ont été testés et invalidés empiriquement sur ce jeu_donnees — cf.
# docs/model-decisions.md. L'horizon unique reste la cible active.
HEURES_HORIZON_EVENEMENT = 120
PAS_HORIZON_EVENEMENT = HEURES_HORIZON_EVENEMENT * 2  # pas de 30 min

# Budget de fausses alertes : combien de dérangements par machine et par jour
# l'équipe de maintenance accepte pour une alerte qui ne mène à rien. Remplace le
# F2 comme objectif de sélection du seuil (point 17/24) — cf. docs/experiments.md
# pour la courbe recall/FA mesurée qui justifie cette valeur.
BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR = 0.2
GRILLE_SEUILS = np.arange(0.005, 0.50, 0.005)


class PipelineModele:
    def __init__(self):
        self.config = charger_config()
        self.colonne_cible = self.config["jeu_donnees"]["colonne_cible"]
        self.colonnes_features = self.config["jeu_donnees"]["features"]
        uri_suivi = os.getenv("MLFLOW_TRACKING_URI", self.config["mlflow"]["uri_suivi"])
        mlflow.set_tracking_uri(uri_suivi)
        self.client = mlflow.MlflowClient()

    def _split_temporel(self, df: pd.DataFrame):
        """
        Split temporel à 3 voies : train (< DATE_COUPURE_VALIDATION), validation
        ([DATE_COUPURE_VALIDATION, DATE_COUPURE) — sert uniquement à choisir le seuil),
        test (>= DATE_COUPURE — jamais vu avant le rapport final). Le seuil est choisi
        sur la validation, pas sur le test, pour éviter le biais optimiste d'un seuil
        ajusté sur les données mêmes qui servent à le rapporter.
        """
        if "timestamp" not in df.columns:
            raise ValueError("La colonne 'timestamp' est requise pour le split temporel.")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        train = df[df["timestamp"] < DATE_COUPURE_VALIDATION]
        validation = df[(df["timestamp"] >= DATE_COUPURE_VALIDATION) & (df["timestamp"] < DATE_COUPURE)]
        test = df[df["timestamp"] >= DATE_COUPURE]
        logger.info(
            f"Split temporel - Train: {len(train):,} (< {DATE_COUPURE_VALIDATION}) | "
            f"Validation: {len(validation):,} ([{DATE_COUPURE_VALIDATION}, {DATE_COUPURE})) | "
            f"Test: {len(test):,} (>= {DATE_COUPURE})"
        )
        return (
            train[self.colonnes_features],
            train[self.colonne_cible],
            validation[self.colonnes_features],
            validation[self.colonne_cible],
            test[self.colonnes_features],
            test[self.colonne_cible],
        )

    def _loguer_matrice_confusion(self, y_true, y_predit):
        cm = confusion_matrix(y_true, y_predit)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        ax.set_title("Confusion Matrix (seuil recall événementiel @ budget FA)")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, bbox_inches="tight", dpi=100)
            mlflow.log_artifact(tmp.name, artifact_path="plots")
        plt.close(fig)

    def _loguer_importances_variables(self, pipeline_complet: Pipeline):
        try:
            lgbm = pipeline_complet.named_steps["classifier"]
            importances = lgbm.feature_importances_.tolist()
            fi_dict = {str(i): float(v) for i, v in enumerate(importances)}
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
                json.dump(fi_dict, tmp)
                mlflow.log_artifact(tmp.name, artifact_path="feature_importances")
        except AttributeError:
            logger.warning("feature_importances_ non disponible pour ce modèle.")

    def _promouvoir_si_meilleur(self, nom_modele: str, nouvelle_version: str, nouveau_score: float) -> bool:
        """
        Gate de promotion : la nouvelle version ne devient '@champion' que si elle égale
        ou dépasse le champion actuel sur METRIQUE_PROMOTION. Sans champion existant
        (premier entraînement), la promotion est automatique. Évite qu'un ré-entraînement
        dégradé ne remplace silencieusement le modèle servi en production.
        """
        try:
            champion = self.client.get_model_version_by_alias(nom_modele, ALIAS_CHAMPION)
            score_champion = self.client.get_run(champion.run_id).data.metrics.get(METRIQUE_PROMOTION, -1.0)
        except Exception:
            champion, score_champion = None, -1.0

        if nouveau_score >= score_champion:
            self.client.set_registered_model_alias(nom_modele, ALIAS_CHAMPION, nouvelle_version)
            logger.info(
                f"Version {nouvelle_version} promue @{ALIAS_CHAMPION} "
                f"({METRIQUE_PROMOTION}={nouveau_score:.4f} >= {score_champion:.4f})"
            )
            return True

        logger.warning(
            f"Version {nouvelle_version} ({METRIQUE_PROMOTION}={nouveau_score:.4f}) n'a pas dépassé le champion actuel "
            f"(version {champion.version}, {METRIQUE_PROMOTION}={score_champion:.4f}) — champion conservé."
        )
        return False

    def lancer_entrainement(self, model, train_df: pd.DataFrame, nom_modele: str, hyperparametres: dict):
        """
        Pipeline d'entraînement complet : split temporel à 3 voies - fit - seuil choisi
        sur validation (recall événementiel sous budget de fausses alertes) - métriques
        rapportées sur test - MLflow.
        """
        mlflow.set_experiment(self.config["mlflow"]["nom_experience"])
        logger.info("Début du pipeline d'entraînement...")

        X_train, y_train, X_val, y_val, X_test, y_test = self._split_temporel(train_df)
        logger.info(f"Pannes train:      {y_train.sum():,} / {len(y_train):,} ({y_train.mean() * 100:.2f}%)")
        logger.info(f"Pannes validation: {y_val.sum():,} / {len(y_val):,} ({y_val.mean() * 100:.2f}%)")
        logger.info(f"Pannes test:       {y_test.sum():,} / {len(y_test):,} ({y_test.mean() * 100:.2f}%)")

        with mlflow.start_run(run_name=f"{nom_modele}_Training"):
            preprocessor = Preprocesseur()
            pipeline_complet = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

            logger.info("Entraînement du modèle...")
            pipeline_complet.fit(X_train, y_train)
            logger.info("Entraînement terminé.")

            probas_brutes_test = pipeline_complet.predict_proba(X_test)[:, 1]

            # Calibration isotonique des probabilités, ajustée sur la validation —
            # remplace le recentrage logit ad hoc de Predicteur._normalize_probability
            # (qui ne faisait que déplacer artificiellement le seuil sur 50 %, sans
            # rendre les probabilités elles-mêmes interprétables). FrozenEstimator :
            # le pipeline est déjà entraîné, seule la calibration doit être fittée
            # (point 22).
            calibreur = CalibratedClassifierCV(FrozenEstimator(pipeline_complet), method="isotonic")
            calibreur.fit(X_val, y_val)

            probas_validation = calibreur.predict_proba(X_val)[:, 1]
            probas_test = calibreur.predict_proba(X_test)[:, 1]

            brier_brut = brier_score_loss(y_test, probas_brutes_test)
            brier_calibre = brier_score_loss(y_test, probas_test)
            logger.info(f"Brier score (test) — brut: {brier_brut:.4f} | calibré: {brier_calibre:.4f}")

            # Seuil choisi sur la validation, jamais sur le test (point 16).
            meta_validation = train_df.loc[X_val.index, ["id_machine", "timestamp", "panne", "mode_panne"]]
            seuil_optimal, metriques_evenement_validation = trouver_seuil_pour_budget_fa(
                meta_validation, probas_validation, GRILLE_SEUILS, PAS_HORIZON_EVENEMENT, BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR
            )
            logger.info(
                f"Seuil retenu (validation) : {seuil_optimal:.3f} — "
                f"recall_evenementiel={metriques_evenement_validation['recall_evenementiel']:.3f}, "
                f"FA/machine/jour={metriques_evenement_validation['taux_fausses_alertes_par_machine_jour']:.3f}"
            )

            # Métriques rapportées sur le test, jamais utilisé pour choisir le seuil.
            meta_test = train_df.loc[X_test.index, ["id_machine", "timestamp", "panne", "mode_panne"]]
            metriques_evenement_test = metriques_evenementielles(meta_test, probas_test, seuil_optimal, PAS_HORIZON_EVENEMENT)
            y_predit = (probas_test >= seuil_optimal).astype(int)

            metrics = {
                "recall_evenementiel": metriques_evenement_test["recall_evenementiel"],
                "taux_fausses_alertes_par_machine_jour": metriques_evenement_test["taux_fausses_alertes_par_machine_jour"],
                "n_failures_test": metriques_evenement_test["nb_pannes"],
                "n_caught_test": metriques_evenement_test["nb_attrapees"],
                "n_false_alarm_episodes_test": metriques_evenement_test["nb_episodes_fausses_alertes"],
                "pr_auc": average_precision_score(y_test, probas_test),
                "roc_auc": roc_auc_score(y_test, probas_test),
                "precision": precision_score(y_test, y_predit, average="binary", pos_label=1, zero_division=0),
                "recall": recall_score(y_test, y_predit, average="binary", pos_label=1, zero_division=0),
                "f1_score": f1_score(y_test, y_predit, average="binary", pos_label=1, zero_division=0),
                "brier_score_raw": brier_brut,
                "brier_score_calibrated": brier_calibre,
                "failure_rate_train": float(y_train.mean()),
                "failure_rate_validation": float(y_val.mean()),
                "failure_rate_test": float(y_test.mean()),
            }

            mlflow.log_params(hyperparametres)
            mlflow.log_param("nom_jeu_donnees", self.config["jeu_donnees"]["name"])
            mlflow.log_param("validation_split_date", DATE_COUPURE_VALIDATION)
            mlflow.log_param("split_date", DATE_COUPURE)
            mlflow.log_param("seuil_optimal", round(seuil_optimal, 4))
            mlflow.log_param("false_alarm_budget_per_machine_day", BUDGET_FAUSSES_ALERTES_PAR_MACHINE_JOUR)
            mlflow.log_param("event_horizon_hours", HEURES_HORIZON_EVENEMENT)
            mlflow.log_param("n_train", len(y_train))
            mlflow.log_param("n_validation", len(y_val))
            mlflow.log_param("n_test", len(y_test))
            mlflow.log_metrics(metrics)

            self._loguer_matrice_confusion(y_test.values, y_predit)
            self._loguer_importances_variables(pipeline_complet)

            # C'est le pipeline CALIBRÉ qui est sauvegardé/loggé — predict_proba()
            # dessus renvoie déjà des probabilités calibrées, plus besoin du
            # recentrage logit ad hoc côté service (cf. src/xpredict.py).
            info_modele = mlflow.sklearn.log_model(
                sk_model=calibreur,
                name="model_pipeline",
                registered_model_name=nom_modele,
            )

            promu = self._promouvoir_si_meilleur(nom_modele, info_modele.registered_model_version, metrics[METRIQUE_PROMOTION])

            if promu:
                os.makedirs("models", exist_ok=True)
                joblib.dump(calibreur, "models/model_pipeline.pkl")
                with open(CHEMIN_SEUIL, "w") as f:
                    json.dump({"threshold": seuil_optimal}, f)
                logger.info("Pipeline sauvegardé : models/model_pipeline.pkl")
                logger.info(f"Seuil optimal sauvegardé : {CHEMIN_SEUIL}")
            else:
                logger.info("Fallback local non mis à jour (version non promue).")

            for name, val in metrics.items():
                print(f"{name}: {val}")
