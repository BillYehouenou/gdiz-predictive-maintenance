import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib

from src.config_loader import load_config
from src.preprocessor import Preprocessor

# uv run mlflow ui --backend-store-uri sqlite:///mlflow.db


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPipeline:
    def __init__(self):
        """
        Initialise le moteur d'entraînement avec la configuration yaml.
        """
        self.config = load_config()
        self.target_col = self.config['dataset']['target_column']
        self.features_cols = self.config['dataset']['features']
        
    def run_training(self, model, train_df: pd.DataFrame, model_name: str, hyperparameters: dict):
        """
        Pipeline d'entrainement complet : de la préparation des données à l'enregistrement du modèle dans MLflow et localement.
        """
        # Configuration de l'expérience MLflow
        mlflow.set_experiment("GDIZ_Predictive_Maintenance")

        logger.info("Début du pipeline d'entraînement...")
        
        with mlflow.start_run(run_name=f"{model_name}_Training"):

            # Séparation Features et Cible 
            X = train_df[self.features_cols]
            y = train_df[self.target_col]

            # Split Train/Test        
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"Données splittées. Train: {X_train.shape[0]} lignes, Test: {X_test.shape[0]} lignes.")

            # Pipeline complet (préprocesseur + modèle)
            preprocessor = Preprocessor()
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Entraînement du pipeline complet
            logger.info("Entraînement du modèle...")
            full_pipeline.fit(X_train, y_train)
            logger.info("Entraînement terminé avec succès !")

            # Évaluation et calcul des métriques
            y_pred = full_pipeline.predict(X_test)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='binary', pos_label=1),
                "recall": recall_score(y_test, y_pred, average='binary', pos_label=1),
                "f1_score": f1_score(y_test, y_pred, average='binary', pos_label=1)
            }

            # Logging dans MLflow (Paramètres, Métriques et Artéfacts)
            logger.info("Enregistrement des données dans MLflow...")
            
            # Enregistrement des paramètres de configuration et du modèle
            mlflow.log_params(hyperparameters)
            mlflow.log_param("dataset_name", self.config['dataset']['name'])
            mlflow.log_param("test_size", 0.2)
            
            # Enregistrement des métriques de performance
            mlflow.log_metrics(metrics)
            
            # Enregistrement du pipeline Scikit-Learn complet directement dans le registre MLflow
            mlflow.sklearn.log_model(
                sk_model=full_pipeline, 
                name="model_pipeline",
                registered_model_name=model_name
            )
            
            # Sauvegarde locale du pipeline 
            os.makedirs("models", exist_ok=True)
            local_model_path = "models/model_pipeline.pkl"
            joblib.dump(full_pipeline, local_model_path)
            logger.info(f"Pipeline complet sauvegardé localement dans : {local_model_path}")
            
            # Affichage des resultats
            for metric_name, val in metrics.items():
                print(f"{metric_name.capitalize()}: {val:.4f}")