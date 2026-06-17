import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.configloader import load_config

logger = logging.getLogger(__name__)

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Initialise le préprocesseur avec la config YAML.
        """
        self.config = load_config()
        self.is_fitted = False
        self.pipeline = None

        # Récupération de la liste des features de la config
        self.all_features = self.config["dataset"]["features"]
        self.categorical_features = ["benin_season", "machine_type"]
        self.numeric_features = [col for col in self.all_features if col not in self.categorical_features]

        logger.info(f"Features numériques identifiées : {self.numeric_features}")
        logger.info(f"Features catégorielles identifiées : {self.categorical_features}")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Configure les étapes de transformation sur les données d'entraînement.
        """
        logger.info("Fitting du pipeline de preprocessing en cours...")

        # 1. Transformation des variables numériques (Imputation + Standardisation)
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

        # 2. Transformation des variables catégorielles (Imputation + Encodage)
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        # Assemblage des deux pipelines
        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ],
            remainder="drop",  # Supprime automatiquement autres colonnes
        )

        existing_features = [col for col in self.all_features if col in X.columns]
        self.pipeline.fit(X[existing_features])

        self.is_fitted = True
        logger.info("Pipeline de preprocessing configuré avec succès.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applique les transformations configurées sur un DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("Le preprocessor doit être 'fitted' avant de pouvoir transformer des données.")

        logger.info("Transformation des données en cours...")
        existing_features = [col for col in self.all_features if col in X.columns]
        result = self.pipeline.transform(X[existing_features])
        feature_names = self.pipeline.get_feature_names_out()
        result = pd.DataFrame(result, columns=feature_names, index=X.index)
        logger.info("Transformation des données terminée.")
        return result

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """
        Combine le fit et le transform.
        """
        self.fit(X)
        return self.transform(X)
