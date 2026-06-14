"""Entrypoint d'entraînement — à lancer via : uv run python scripts/run_training.py"""

from sklearn.ensemble import RandomForestClassifier
from src.dataloader import DataLoader
from src.train import ModelPipeline

if __name__ == "__main__":
    dl = DataLoader()
    df = dl.load_raw_data()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    hyperparams = {"n_estimators": 100, "random_state": 42}

    mp = ModelPipeline()
    mp.run_training(model, df, "RF_Baseline", hyperparams)
