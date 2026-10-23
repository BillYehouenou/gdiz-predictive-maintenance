"""
Génère le dataset GDIZ : météo réelle (Open-Meteo) --> physique des machines --> observation IoT --> DuckDB.
Usage : python -m simulation.generer
"""

import logging
from datetime import timedelta

import pandas as pd

from simulation.meteo import charger_meteo
from simulation.observation import AgregateurDonnees
from simulation.physique import ConfigDonneesSynthetiques, EvolueurEtatMachine, GenerateurContexteExogene

logger = logging.getLogger(__name__)


def generer(config: ConfigDonneesSynthetiques | None = None) -> pd.DataFrame:
    config = config or ConfigDonneesSynthetiques()
    debut = config.date_debut.date()
    fin = debut + timedelta(days=config.duree_jours - 1)

    meteo_horaire = charger_meteo(config.fichier_duckdb, debut, fin)
    df_global = GenerateurContexteExogene(config).generer_contexte_global(meteo_horaire)

    # (L, M, H) : 17/17/16 pour 50 machines.
    types = ["L", "M", "H"]
    df_machines = pd.concat(
        [EvolueurEtatMachine(mid, types[i % 3], df_global, config).evoluer_etats() for i, mid in enumerate(config.ids_machines)],
        ignore_index=True,
    )
    df = df_machines.merge(df_global, on="timestamp", how="left", validate="many_to_one")

    agregateur = AgregateurDonnees(config)
    df = agregateur.injecter_donnees_manquantes(df)
    agregateur.inserer_duckdb(df, config.fichier_duckdb)

    logger.info(
        f"Dataset généré : {config.nb_machines} machines, {config.duree_jours} jours, {config.intervalle_minutes} min/pas, "
        f"{int(df['panne'].sum())} pannes."
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
    generer()
