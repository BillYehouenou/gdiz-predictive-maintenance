import pandas as pd
import logging
import pathlib
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path: str):
        """
        Initialise le chargeur de données.
        data_path: Chemin vers le dossier contenant les fichiers CSV ou Parquet.
        """
        self.data_path = pathlib.Path(data_path)
        
    def load_raw_data(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Charge un fichier CSV ou Parquet depuis le dossier data/raw.
        """
        file_full_path = self.data_path / 'raw' / filename
        
        try:
            logger.info(f"Tentative de chargement des données depuis : {file_full_path}")
            
            if not file_full_path.exists():
                raise FileNotFoundError(f"Le fichier {filename} est introuvable dans {file_full_path}")
            
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
    loader = DataLoader(data_path="data")
    data = loader.load_raw_data("gdiz_dataset.csv")