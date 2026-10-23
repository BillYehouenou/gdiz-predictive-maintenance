import logging
import pathlib

import duckdb
import pandas as pd

from src.configloader import charger_config
from src.features import enrichir_df_entrainement
from src.validation import valider_donnees_brutes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChargeurDonnees:
    def __init__(self):
        self.config = charger_config()
        self.chemin_base = pathlib.Path(self.config["chemins"]["chemin_donnees_brutes"]) / self.config["jeu_donnees"]["base"]
        self.table = self.config["jeu_donnees"]["table"]

    def charger_donnees_brutes(self) -> pd.DataFrame:
        """Charge la télémétrie depuis DuckDB, calcule les features temporelles, retourne le DataFrame enrichi."""
        if not self.chemin_base.exists():
            raise FileNotFoundError(f"Base introuvable : {self.chemin_base}")

        with duckdb.connect(str(self.chemin_base), read_only=True) as conn:
            df = conn.execute(f"SELECT * FROM {self.table}").fetchdf()

        if df.empty:
            logger.warning(f"{self.table} est vide.")
            return df

        logger.info(f"Données brutes chargées : {df.shape[0]:,} lignes, {df.shape[1]} colonnes.")
        df = valider_donnees_brutes(df)
        df = enrichir_df_entrainement(df)
        logger.info(f"Dataset enrichi avec features temporelles → {df.shape[1]} colonnes totales.")
        return df
