"""Entrypoint d'entraînement — uv run python scripts/run_training.py"""

from lightgbm import LGBMClassifier
from src.dataloader import DataLoader
from src.train import ModelPipeline

if __name__ == "__main__":
    dl = DataLoader()
    df = dl.load_raw_data()

    # Calibré pour target_h120_predictable (cf. memory project_ml_stack) : sans
    # class_weight (le rééquilibrage "balanced" dégradait le F2 sur ce label),
    # peu de feuilles + forte régularisation pour éviter l'overfit sur un signal rare.
    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=0.3,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    hyperparams = {
        "model_type": "LGBMClassifier",
        "n_estimators": 150,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_child_samples": 80,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "split_strategy": "temporal",
    }

    mp = ModelPipeline()
    mp.run_training(model, df, "LGBM_Predictive_Maintenance", hyperparams)
