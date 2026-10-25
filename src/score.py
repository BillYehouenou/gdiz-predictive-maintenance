"""
Score les 60 derniers jours du parc avec le @champion et écrit dans DuckDB la table `predictions`
(risque, niveau et cause probable), lue par l'API. Usage : python -m src.score (après chaque entraînement)
"""

import logging

import duckdb
import numpy as np
import pandas as pd

from src.config import CHEMIN_DB, JOURS_PREDICTIONS, TABLE_PREDICTIONS
from src.donnees import charger
from src.explain import ExplicateurCause
from src.predicteur import Predicteur

logger = logging.getLogger(__name__)


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
        causes = explicateur.cause_probable(en_alerte)
        predictions.loc[en_alerte.index, "cause_principale"] = [c["cause_principale"] for c in causes]
    return predictions


def main():
    predicteur = Predicteur()
    df = charger()  # features calculées sur tout l'historique, puis seule la fenêtre affichée est scorée
    df = df[df["timestamp"] > df["timestamp"].max() - pd.Timedelta(days=JOURS_PREDICTIONS)].reset_index(drop=True)
    predictions = scorer(df, predicteur, ExplicateurCause(predicteur.modele))
    with duckdb.connect(str(CHEMIN_DB)) as conn:
        conn.register("df_predictions", predictions)
        conn.execute(f"CREATE OR REPLACE TABLE {TABLE_PREDICTIONS} AS SELECT * FROM df_predictions")
        conn.execute(f"CREATE INDEX idx_predictions ON {TABLE_PREDICTIONS}(id_machine, timestamp)")
    alertes = predictions.dropna(subset=["cause_principale"])
    logger.info(f"{len(predictions):,} prédictions écrites (modèle v{predicteur.version}), {len(alertes)} machines en alerte.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
