"""
Score tout le parc avec le @champion et écrit dans DuckDB :
- `predictions` : risque, niveau et cause probable, lus par l'API et le dashboard ;
- `reglage` : sur 2025, ce que coûte et rapporte chaque niveau de seuil (fausses alertes face aux pannes annoncées).
Usage : python -m src.score (après chaque entraînement)
"""

import logging

import duckdb
import numpy as np
import pandas as pd

from src.config import CHEMIN_DB, TABLE_PREDICTIONS, TABLE_REGLAGE
from src.donnees import DATE_TEST, MODES_PANNE_PREDICTIBLES, charger
from src.evaluation import PAS_HORIZON, metriques_evenementielles
from src.explain import ExplicateurCause
from src.predicteur import Predicteur
from src.usine import HEURES_ARRET_EVITEES, HEURES_FAUSSE_ALERTE, TAUX_TECHNICIEN, cout_heure_arret

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


def courbe_reglage(test: pd.DataFrame, probas: np.ndarray, seuil: float, nb_points: int = 20) -> pd.DataFrame:
    """
    Pour une série de seuils : fausses alertes par machine-jour, part des pannes annoncées, alertes par jour et bilan net.
    Bilan = pannes annoncées × arrêt évité − fausses alertes × temps technicien (hypothèses de src/usine.py).
    """
    predictibles = test[(test["panne"] == 1) & test["mode_panne"].isin(MODES_PANNE_PREDICTIBLES)]
    gain_par_panne = HEURES_ARRET_EVITEES * float(np.mean([cout_heure_arret(g) for g in predictibles["type_machine"]]))
    cout_fausse_alerte = HEURES_FAUSSE_ALERTE * TAUX_TECHNICIEN
    jours = (test["timestamp"].max() - test["timestamp"].min()) / pd.Timedelta("1D")
    lignes = []
    for s in sorted({*np.quantile(probas, np.linspace(0.80, 0.995, nb_points)).tolist(), seuil}):
        m = metriques_evenementielles(test, probas, s, PAS_HORIZON)
        lignes.append(
            {
                "seuil": s,
                "fausses_alertes_machine_jour": m["taux_fausses_alertes_par_machine_jour"],
                "part_pannes_annoncees": m["recall_evenementiel"],
                "alertes_jour": m["nb_episodes"] / jours,
                "bilan": m["nb_attrapees"] * gain_par_panne - m["nb_episodes_fausses_alertes"] * cout_fausse_alerte,
                "preavis_median_h": float(np.median(m["preavis_heures"])) if m["preavis_heures"] else None,
                "actuel": s == seuil,
            }
        )
    return pd.DataFrame(lignes)


def main():
    predicteur = Predicteur()
    df = charger()
    predictions = scorer(df, predicteur, ExplicateurCause(predicteur.modele))
    test = df["timestamp"] >= DATE_TEST
    reglage = courbe_reglage(df[test], predictions.loc[test, "probabilite_panne"].to_numpy(), predicteur.threshold)
    with duckdb.connect(str(CHEMIN_DB)) as conn:
        conn.register("df_predictions", predictions)
        conn.execute(f"CREATE OR REPLACE TABLE {TABLE_PREDICTIONS} AS SELECT * FROM df_predictions")
        conn.execute(f"CREATE INDEX idx_predictions ON {TABLE_PREDICTIONS}(id_machine, timestamp)")
        conn.register("df_reglage", reglage)
        conn.execute(f"CREATE OR REPLACE TABLE {TABLE_REGLAGE} AS SELECT * FROM df_reglage")
    alertes = predictions.dropna(subset=["cause_principale"])
    logger.info(f"{len(predictions):,} prédictions écrites (modèle v{predicteur.version}), {len(alertes)} machines en alerte.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
