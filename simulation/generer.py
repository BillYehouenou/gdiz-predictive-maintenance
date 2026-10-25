"""
Génère le dataset GDIZ : météo réelle (Open-Meteo) --> physique des machines --> défaillances des capteurs IoT
--> DuckDB. Usage : python -m simulation.generer
"""

import logging
import pathlib
from datetime import timedelta

import duckdb
import numpy as np
import pandas as pd

from simulation.meteo import charger_meteo
from simulation.physique import ConfigDonneesSynthetiques, EvolueurEtatMachine, GenerateurContexteExogene
from src.config import TABLE

logger = logging.getLogger(__name__)

COLONNES_CAPTEURS = [
    "temperature_process",
    "vitesse_rotation",
    "couple",
    "vibration",
    "temperature_ambiante",
    "humidite",
    "concentration_poussiere",
]


def _rafales(rng, valeurs: np.ndarray, reglage: tuple, mode: str, derive_max: float = 0.0) -> np.ndarray:
    """
    Événements indépendants de durée log-normale sur la série d'un capteur : « nan » (paquets perdus), « bloque »
    (valeur figée) ou « derive » (biais silencieux qui croît jusqu'à derive_max, de signe aléatoire).
    """
    probabilite, pas_moyens, sigma, plage = reglage
    sortie = valeurs.copy()
    for s in np.flatnonzero(rng.random(len(valeurs)) < probabilite).tolist():
        fin = min(s + int(np.ceil(np.clip(rng.lognormal(np.log(pas_moyens), sigma), *plage))), len(valeurs))
        if mode == "derive":
            sortie[s:fin] += (1.0 if rng.random() < 0.5 else -1.0) * derive_max * np.linspace(0.0, 1.0, fin - s)
        else:
            sortie[s:fin] = np.nan if mode == "nan" else valeurs[s]
    return sortie


def defaillances_capteurs(df: pd.DataFrame, config: ConfigDonneesSynthetiques) -> pd.DataFrame:
    """Dérives, capteurs figés et rafales de valeurs manquantes, tirés indépendamment par (machine, capteur)."""
    rng = np.random.default_rng(config.graine_globale)
    df = df.copy()
    colonnes = [c for c in COLONNES_CAPTEURS if c in df.columns]
    donnees = df[colonnes].to_numpy(dtype=float, copy=True)
    for positions in df.groupby("id_machine", sort=False).indices.values():
        for k in range(len(colonnes)):
            v = donnees[positions, k]
            echelle = float(np.nanstd(v)) or 1.0
            v = _rafales(rng, v, config.derive_capteur, "derive", echelle * config.pct_max_derive_capteur)
            v = _rafales(rng, v, config.capteur_bloque, "bloque")
            donnees[positions, k] = _rafales(rng, v, config.rafale_nan, "nan")
    df[colonnes] = donnees
    return df


def ecrire(df: pd.DataFrame, chemin_db: pathlib.Path):
    chemin_db.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(chemin_db)) as conn:
        conn.register("df_telemetrie", df)
        conn.execute(f"CREATE OR REPLACE TABLE {TABLE} AS SELECT * FROM df_telemetrie")
        conn.execute(f"CREATE INDEX idx_machine_time ON {TABLE}(id_machine, timestamp)")


def generer(config: ConfigDonneesSynthetiques | None = None) -> pd.DataFrame:
    config = config or ConfigDonneesSynthetiques()
    debut = config.date_debut.date()
    meteo = charger_meteo(config.fichier_duckdb, debut, debut + timedelta(days=config.duree_jours - 1))
    df_global = GenerateurContexteExogene(config).generer_contexte_global(meteo)
    gammes = ["L", "M", "H"]  # 17 / 17 / 16 machines
    machines = [
        EvolueurEtatMachine(m, gammes[i % 3], df_global, config).evoluer_etats() for i, m in enumerate(config.ids_machines)
    ]
    df = pd.concat(machines, ignore_index=True).merge(df_global, on="timestamp", how="left", validate="many_to_one")
    df = defaillances_capteurs(df, config)
    ecrire(df, config.fichier_duckdb)
    logger.info(f"Dataset généré : {config.nb_machines} machines, {config.duree_jours} jours, {int(df['panne'].sum())} pannes.")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generer()
