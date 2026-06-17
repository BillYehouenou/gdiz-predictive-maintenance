import pathlib
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def load_config(config_name: str = "config/config.yaml"):
    """Charge le fichier de configuration YAML."""
    config_path = PROJECT_ROOT / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable avec le chemin : {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # On rend les chemins de la config absolus dynamiquement pour que data_loader ne se perde jamais
    config["paths"]["raw_data_path"] = str(PROJECT_ROOT / config["paths"]["raw_data_path"])
    return config