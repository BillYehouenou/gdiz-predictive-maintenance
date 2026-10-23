"""
Features glissantes par machine, sur des fenêtres de durée.
Une seule fonction, utilisée à l'entraînement, par le dashboard et par l'API.
"""

import pandas as pd

from src.config import FEATURES_BRUTES, PAS

CAPTEURS = ["vibration", "temperature_process", "ecart_thermique", "couple", "concentration_poussiere"]
FEATURES_GLISSANTES = [f"{c}_{stat}" for c in CAPTEURS for stat in ("moyenne_24h", "max_24h", "ecart_type_24h", "hausse_4h")]
FEATURES = FEATURES_BRUTES + ["ecart_thermique"] + FEATURES_GLISSANTES
TYPES_MACHINE = ["L", "M", "H"]

HISTORIQUE = pd.Timedelta("24h")  # plus longue fenêtre : historique à fournir pour prédire un point
LIGNES_HISTORIQUE = int(HISTORIQUE / PAS)


def calculer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features glissantes. Retourne le DataFrame trié par (id_machine, timestamp), index réinitialisé."""
    df = df.sort_values(["id_machine", "timestamp"]).reset_index(drop=True)
    df["ecart_thermique"] = df["temperature_process"] - df["temperature_ambiante"]

    groupes = df.set_index("timestamp").groupby("id_machine", sort=False)[CAPTEURS]
    fenetre_24h = groupes.rolling(HISTORIQUE, min_periods=1)
    moyenne, maximum, ecart_type = fenetre_24h.mean(), fenetre_24h.max(), fenetre_24h.std()
    minimum_4h = groupes.rolling("4h", min_periods=1).min()

    for c in CAPTEURS:
        df[f"{c}_moyenne_24h"] = moyenne[c].to_numpy()
        df[f"{c}_max_24h"] = maximum[c].to_numpy()
        df[f"{c}_ecart_type_24h"] = ecart_type[c].to_numpy()
        df[f"{c}_hausse_4h"] = df[c] - minimum_4h[c].to_numpy()
    return df


def preparer_X(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes du modèle. LightGBM gère nativement les NaN et la catégorie type_machine."""
    X = df[FEATURES].copy()
    X["type_machine"] = pd.Categorical(X["type_machine"], categories=TYPES_MACHINE)
    return X
