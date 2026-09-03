"""Entrypoint d'entraînement — uv run python scripts/run_training.py"""

from lightgbm import LGBMClassifier

from src.dataloader import ChargeurDonnees
from src.train import PipelineModele

if __name__ == "__main__":
    chargeur = ChargeurDonnees()
    df = chargeur.charger_donnees_brutes()

    # Calibré pour cible_h120_predictible : sans class_weight,
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

    hyperparametres = {
        "type_modele": "LGBMClassifier",
        "n_estimators": 150,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_child_samples": 80,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "strategie_split": "temporal",
    }

    pipeline = PipelineModele()
    pipeline.lancer_entrainement(model, df, "LGBM_Predictive_Maintenance", hyperparametres)
