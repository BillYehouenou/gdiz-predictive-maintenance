import logging
import pathlib
import pandas as pd
from src.configloader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        """
        Initialise le chargeur de données en chargeant la configuration et en définissant les chemins d'accès.
        """
        self.config = load_config()
        self.raw_path = pathlib.Path(self.config["paths"]["raw_data_path"])

    def load_raw_data(self) -> pd.DataFrame:
        """
        Charge un fichier CSV ou Parquet.
        """
        filename = self.config["dataset"]["name"]
        file_full_path = self.raw_path / filename

        try:
            logger.info("Tentative de chargement des données")

            if not file_full_path.exists():
                raise FileNotFoundError(f"Le fichier {filename} est introuvable dans {self.raw_path}")

            if filename.endswith(".csv"):
                df = pd.read_csv(file_full_path)
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(file_full_path)
            else:
                raise ValueError(f"Format de fichier non supporté : {filename}")

            if df.empty:
                logger.warning(f"Le fichier {filename} est vide.")
                return None

            logger.info(f"Chargement réussi : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
            return df

        except Exception as e:
            logger.error(f"Erreur lors du chargement des données : {e}")
            return None

# Test
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_raw_data()