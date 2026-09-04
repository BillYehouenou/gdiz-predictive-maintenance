import logging

import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

logger = logging.getLogger(__name__)

SCHEMA_DONNEES_BRUTES = pa.DataFrameSchema(
    {
        "type_machine": Column(str, Check.isin(["L", "M", "H"]), nullable=True, required=False),
        "panne": Column(int, Check.isin([0, 1]), nullable=True, required=False),
        "saison_benin": Column(int, Check.isin([0, 1, 2, 3]), nullable=True, required=False),
        "indicateur_pluie": Column(int, Check.isin([0, 1]), nullable=True, required=False),
        "indicateur_coupure": Column(int, Check.isin([0, 1]), nullable=True, required=False),
        "temperature_process": Column(float, Check.in_range(-50, 200), nullable=True, required=False),
        "temperature_ambiante": Column(float, Check.in_range(-20, 60), nullable=True, required=False),
        "vitesse_rotation": Column(float, Check.in_range(-1000, 5000), nullable=True, required=False),
        "couple": Column(float, Check.in_range(-100, 500), nullable=True, required=False),
        "vibration": Column(float, Check.in_range(-10, 50), nullable=True, required=False),
        "usure_outil": Column(float, Check.in_range(-50, 150), nullable=True, required=False),
        "humidite": Column(float, Check.in_range(-20, 150), nullable=True, required=False),
        "concentration_poussiere": Column(float, Check.in_range(-100, 1000), nullable=True, required=False),
        "niveau_tension": Column(float, Check.in_range(0, 400), nullable=True, required=False),
        "stabilite_tension": Column(float, Check.in_range(-50, 150), nullable=True, required=False),
        "niveau_activite": Column(float, Check.in_range(-0.5, 1.5), nullable=True, required=False),
    },
    strict=False,
    coerce=False,
)


def valider_donnees_brutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide les colonnes présentes par rapport aux schémas et logue unrésumé des anomalies.
    Ne bloque jamais le chargement, on signale juste un problème de qualité de données.
    """
    try:
        SCHEMA_DONNEES_BRUTES.validate(df, lazy=True)
        logger.info("Validation des bornes physiques à l'ingestion : aucune anomalie.")
    except pa.errors.SchemaErrors as exc:
        resume = exc.failure_cases.groupby("column").size().to_dict()
        logger.warning(f"Anomalies détectées à l'ingestion (bornes physiques dépassées) : {resume}")
    return df
