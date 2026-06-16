"""Entrypoint d'entraînement — uv run python scripts/run_training.py"""

from lightgbm import LGBMClassifier

from src.dataloader import DataLoader
from src.train import ModelPipeline

if __name__ == "__main__":
    dl = DataLoader()
    df = dl.load_raw_data()

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    hyperparams = {
        "model_type": "LGBMClassifier",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
        "split_strategy": "temporal",
    }

    mp = ModelPipeline()
    mp.run_training(model, df, "LGBM_Predictive_Maintenance", hyperparams)
