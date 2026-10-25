"""
Le modèle en service : le @champion du registre MLflow avec son seuil, la cause probable d'une alerte (SHAP) et le
scoring des relevés en marche des 60 derniers jours, écrit dans la table predictions. Usage : python -m src.modele
"""

import logging

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap

from src.config import ALIAS_CHAMPION, CHEMIN_DB, JOURS_PREDICTIONS, NOM_MODELE, TABLE_PREDICTIONS, URI_MLFLOW
from src.donnees import charger, preparer_X

logger = logging.getLogger(__name__)

# Une feature appartient à la famille de sa mesure d'origine (ex. vibration_max_24h -> vibration)
FAMILLES_FEATURES: dict[str, list[str]] = {
    "Usure et entretien": ["heures_depuis_changement_outil", "heures_depuis_maintenance"],
    "Surchauffe": ["temperature_process", "ecart_thermique"],
    "Surcharge mécanique": ["couple", "vibration"],
    "Environnement / réseau": ["concentration_poussiere", "niveau_tension", "indicateur_coupure", "humidite", "precipitation_mm"],
}


class Predicteur:
    """Le @champion et le seuil choisi lors de son entraînement : les deux ne peuvent pas diverger."""

    def __init__(self):
        mlflow.set_tracking_uri(URI_MLFLOW)
        client = mlflow.MlflowClient()
        try:
            version = client.get_model_version_by_alias(NOM_MODELE, ALIAS_CHAMPION)
        except Exception as e:
            raise RuntimeError(f"Aucun @{ALIAS_CHAMPION} dans le registre MLflow ({URI_MLFLOW}) : lancer `make train`.") from e
        self.nom_modele, self.version = NOM_MODELE, str(version.version)
        self.modele = mlflow.sklearn.load_model(f"models:/{NOM_MODELE}@{ALIAS_CHAMPION}")
        self.threshold = float(client.get_run(str(version.run_id)).data.params["seuil_optimal"])

    def predire(self, df: pd.DataFrame) -> pd.DataFrame:
        """df : lignes avec features calculées (src.donnees.calculer_features)."""
        probas = self.modele.predict_proba(preparer_X(df))[:, 1]
        return pd.DataFrame({"prediction": (probas >= self.threshold).astype(int), "probabilite_panne": probas})


def _valeur(v):
    return None if pd.isna(v) else v if isinstance(v, str) else float(v)


def famille_de(feature: str) -> str | None:
    return next(
        (f for f, mesures in FAMILLES_FEATURES.items() if any(feature == m or feature.startswith(f"{m}_") for m in mesures)), None
    )


class ExplicateurCause:
    def __init__(self, modele_calibre):
        # CalibratedClassifierCV(FrozenEstimator(lgbm)) : on explique le LightGBM sous-jacent
        self._explainer = shap.TreeExplainer(modele_calibre.estimator.estimator)

    def cause_probable(self, df: pd.DataFrame) -> list[dict]:
        """Par ligne, la famille dont les contributions vers la panne sont les plus fortes (les autres ne comptent pas)."""
        X = preparer_X(df)
        familles = [famille_de(col) for col in X.columns]
        resultats = []
        for ligne in self._explainer.shap_values(X):
            scores = dict.fromkeys(FAMILLES_FEATURES, 0.0)
            for famille, valeur in zip(familles, ligne):
                if famille:
                    scores[famille] += max(0.0, float(valeur))
            principale, score = max(scores.items(), key=lambda kv: kv[1])
            resultats.append({"cause_principale": principale if score > 0 else "Aucun signal dominant", "score": round(score, 4),
                              "detail": {f: round(s, 4) for f, s in scores.items()}})  # fmt: skip
        return resultats

    def contributions(self, ligne: pd.DataFrame, n: int = 4) -> list[dict]:
        """Les n features qui poussent le plus vers la panne sur une ligne : nom, famille, valeur, contribution."""
        X = preparer_X(ligne)
        top = sorted(zip(X.columns, self._explainer.shap_values(X)[0]), key=lambda kv: kv[1], reverse=True)[:n]
        return [{"feature": f, "famille": famille_de(f), "valeur": _valeur(X.iloc[0][f]), "contribution": round(float(v), 3)}
                for f, v in top if v > 0]  # fmt: skip


def niveaux(probas: np.ndarray, seuil: float) -> np.ndarray:
    """normal sous le seuil, surveillance au-dessus, eleve à partir de 3 fois le seuil."""
    return np.select([probas >= 3 * seuil, probas >= seuil], ["eleve", "surveillance"], "normal")


def scorer(df: pd.DataFrame, predicteur: Predicteur, explicateur: ExplicateurCause) -> pd.DataFrame:
    """df : features calculées, trié par (id_machine, timestamp). Cause probable sur le dernier relevé des machines en alerte."""
    probas = predicteur.predire(df)["probabilite_panne"].to_numpy()
    predictions = df[["id_machine", "timestamp"]].assign(
        probabilite_panne=probas, niveau=niveaux(probas, predicteur.threshold), cause_principale=None
    )
    dernieres = df.groupby("id_machine", sort=False).tail(1)
    en_alerte = dernieres[predictions.loc[dernieres.index, "niveau"] != "normal"]
    if not en_alerte.empty:
        predictions.loc[en_alerte.index, "cause_principale"] = [
            c["cause_principale"] for c in explicateur.cause_probable(en_alerte)
        ]
    return predictions


def main():
    predicteur = Predicteur()
    df = charger()  # features calculées sur tout l'historique, puis seule la fenêtre affichée est scorée
    recent = df["timestamp"] > df["timestamp"].max() - pd.Timedelta(days=JOURS_PREDICTIONS)
    df = df[recent & (df["etat"] == "marche")].reset_index(drop=True)  # une machine arrêtée n'est pas notée
    predictions = scorer(df, predicteur, ExplicateurCause(predicteur.modele))
    with duckdb.connect(str(CHEMIN_DB)) as conn:
        conn.register("df_predictions", predictions)
        conn.execute(f"CREATE OR REPLACE TABLE {TABLE_PREDICTIONS} AS SELECT * FROM df_predictions")
        conn.execute(f"CREATE INDEX idx_predictions ON {TABLE_PREDICTIONS}(id_machine, timestamp)")
    logger.info(f"{len(predictions):,} prédictions écrites (modèle v{predicteur.version}).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
