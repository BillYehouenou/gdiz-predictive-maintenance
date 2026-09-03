import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.configloader import charger_config

logger = logging.getLogger(__name__)


class Preprocesseur(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Initialise le préprocesseur avec la config YAML.
        """
        self.config = charger_config()
        self.est_entraine = False
        self.pipeline = None
        self._types_categoriels: dict[str, pd.CategoricalDtype] = {}

        # Récupération de la liste des features de la config
        self.all_features = self.config["jeu_donnees"]["features"]
        self.categorical_features = ["saison_benin", "type_machine"]
        self.numeric_features = [col for col in self.all_features if col not in self.categorical_features]

        logger.info(f"Features numériques identifiées : {self.numeric_features}")
        logger.info(f"Features catégorielles identifiées : {self.categorical_features}")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Configure les étapes de transformation sur les données d'entraînement.
        """
        logger.info("Fitting du pipeline de preprocessing en cours...")

        # Variables numériques : imputation seule.
        transformateur_numerique = SimpleImputer(strategy="median")

        # Variables catégorielles : encodage ordinal + dtype 'category' plutôt que
        # one-hot. LightGBM gère nativement les colonnes catégorielles
        transformateur_categoriel = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", transformateur_numerique, self.numeric_features),
                ("cat", transformateur_categoriel, self.categorical_features),
            ],
            remainder="drop",  # Supprime automatiquement autres colonnes
        ).set_output(transform="pandas")

        variables_existantes = [col for col in self.all_features if col in X.columns]
        self.pipeline.fit(X[variables_existantes])

        encodeur_ordinal = self.pipeline.named_transformers_["cat"].named_steps["ordinal"]
        self._types_categoriels = {
            f"cat__{col}": pd.CategoricalDtype(categories=list(range(len(categories))) + [-1])
            for col, categories in zip(self.categorical_features, encodeur_ordinal.categories_)
        }

        self.est_entraine = True
        logger.info("Pipeline de preprocessing configuré avec succès.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applique les transformations configurées sur un DataFrame.
        """
        if not self.est_entraine:
            raise ValueError("Le preprocessor doit être 'fitted' avant de pouvoir transformer des données.")

        logger.info("Transformation des données en cours...")
        variables_existantes = [col for col in self.all_features if col in X.columns]
        result = self.pipeline.transform(X[variables_existantes])
        
        for col, dtype in getattr(self, "_types_categoriels", {}).items():
            result[col] = result[col].astype(int).astype(dtype)
        result.index = X[variables_existantes].index
        logger.info("Transformation des données terminée.")
        return result

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Combine le fit et le transform.
        """
        self.fit(X)
        return self.transform(X)
