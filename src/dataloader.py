import logging
import pathlib

import pandas as pd

from src.configloader import charger_config
from src.features import enrichir_df_entrainement

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChargeurDonnees:
    def __init__(self):
        self.config = charger_config()
        self.chemin_brut = pathlib.Path(self.config["chemins"]["chemin_donnees_brutes"])

    def charger_donnees_brutes(self) -> pd.DataFrame:
        """Charge le parquet brut, calcule les features temporelles, retourne le DataFrame enrichi."""
        filename = self.config["jeu_donnees"]["name"]
        chemin_fichier_complet = self.chemin_brut / filename

        if not chemin_fichier_complet.exists():
            raise FileNotFoundError(f"Fichier introuvable : {chemin_fichier_complet}")

        if filename.endswith(".csv"):
            df = pd.read_csv(chemin_fichier_complet)
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(chemin_fichier_complet)
        else:
            raise ValueError(f"Format non supporté : {filename}")

        if df.empty:
            logger.warning(f"{filename} est vide.")
            return df

        logger.info(f"Données brutes chargées : {df.shape[0]:,} lignes, {df.shape[1]} colonnes.")
        df = enrichir_df_entrainement(df)
        logger.info(f"Dataset enrichi avec features temporelles → {df.shape[1]} colonnes totales.")
        return df
