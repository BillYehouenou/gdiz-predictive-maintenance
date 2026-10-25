"""Le @champion du registre MLflow, avec le seuil choisi lors de son entraînement."""

import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import ALIAS_CHAMPION, NOM_MODELE, URI_MLFLOW
from src.features import preparer_X


class Predicteur:
    def __init__(self):
        mlflow.set_tracking_uri(URI_MLFLOW)
        client = mlflow.MlflowClient()
        try:
            version = client.get_model_version_by_alias(NOM_MODELE, ALIAS_CHAMPION)
        except Exception as e:
            raise RuntimeError(f"Aucun @{ALIAS_CHAMPION} dans le registre MLflow ({URI_MLFLOW}) : lancer `make train`.") from e
        self.nom_modele = NOM_MODELE
        self.version = str(version.version)
        self.modele = mlflow.sklearn.load_model(f"models:/{NOM_MODELE}@{ALIAS_CHAMPION}")
        # Le seuil vient du même entraînement que le modèle : les deux ne peuvent pas diverger
        self.threshold = float(client.get_run(str(version.run_id)).data.params["seuil_optimal"])

    def predire(self, df: pd.DataFrame) -> pd.DataFrame:
        """df : lignes avec features calculées (src.features.calculer_features)."""
        probas = self.modele.predict_proba(preparer_X(df))[:, 1]
        return pd.DataFrame({"prediction": (probas >= self.threshold).astype(int), "probabilite_panne": probas})
