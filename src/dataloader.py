import logging
import pathlib
import pandas as pd
from src.configloader import load_config
from src.features import enrich_training_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.config = load_config()
        self.raw_path = pathlib.Path(self.config["paths"]["raw_data_path"])

    def load_raw_data(self) -> pd.DataFrame:
        """Charge le parquet brut, calcule les features temporelles, retourne le DataFrame enrichi."""
        filename = self.config["dataset"]["name"]
        file_full_path = self.raw_path / filename

        if not file_full_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_full_path}")

        if filename.endswith(".csv"):
            df = pd.read_csv(file_full_path)
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(file_full_path)
        else:
            raise ValueError(f"Format non supporté : {filename}")

        if df.empty:
            logger.warning(f"{filename} est vide.")
            return df

        logger.info(f"Données brutes chargées : {df.shape[0]:,} lignes, {df.shape[1]} colonnes.")
        df = enrich_training_df(df)
        logger.info(f"Dataset enrichi avec features temporelles → {df.shape[1]} colonnes totales.")
        return df
