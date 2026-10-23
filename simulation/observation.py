"""
Couche d'observation : ce que les capteurs IoT transmettent réellement (défaillances comprises),
puis persistance dans DuckDB.
"""

import logging
import pathlib

import duckdb
import numpy as np
import pandas as pd

from simulation.physique import ConfigDonneesSynthetiques

logger = logging.getLogger(__name__)

TABLE_TELEMETRIE = "factory_telemetry"
COLONNES_CAPTEURS = [
    "temperature_process",
    "vitesse_rotation",
    "couple",
    "vibration",
    "temperature_ambiante",
    "humidite",
    "concentration_poussiere",
]


class AgregateurDonnees:
    """Applique les défaillances capteurs et persiste la télémétrie."""

    def __init__(self, config: ConfigDonneesSynthetiques | None = None):
        self.config = config or ConfigDonneesSynthetiques()
        self.rng = np.random.default_rng(self.config.graine_globale)

    def _duree(self, pas_moyens: float, sigma: float, plage: tuple) -> int:
        return int(np.ceil(np.clip(self.rng.lognormal(np.log(pas_moyens), sigma), *plage)))

    def _appliquer_rafales(
        self,
        values: np.ndarray,
        start_prob: float,
        duration_mean_steps: float,
        duration_sigma: float,
        duration_range: tuple,
        mode: str,
    ) -> tuple[np.ndarray, int]:
        """
        Applique des rafales indépendantes à une série 1D d'un (machine, capteur) :
        mode="nan" (perte de paquets) ou mode="stuck" (capteur figé à sa valeur de
        début de rafale). Durée tirée d'une loi log-normale.
        """
        n = len(values)
        out = values.copy()
        affected = 0
        for s in np.flatnonzero(self.rng.random(n) < start_prob).tolist():
            end = min(s + self._duree(duration_mean_steps, duration_sigma, duration_range), n)
            out[s:end] = np.nan if mode == "nan" else values[s]
            affected += end - s
        return out, affected

    def _appliquer_derive(self, values: np.ndarray, echelle: float) -> tuple[np.ndarray, int]:
        """
        Dérive de calibration : biais qui croît linéairement (rampe 0 → max) sur la
        durée de l'événement, signe aléatoire, amplitude proportionnelle à l'écart-type
        du capteur. Pas de valeur manquante, un biais silencieux, plus difficile à
        repérer qu'un NaN.
        """
        c = self.config
        n = len(values)
        out = values.copy()
        affected = 0
        for s in np.flatnonzero(self.rng.random(n) < c.probabilite_debut_derive_capteur).tolist():
            end = min(
                s + self._duree(c.pas_moyens_duree_derive_capteur, c.sigma_duree_derive_capteur, c.plage_duree_derive_capteur), n
            )
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            out[s:end] += sign * echelle * c.pct_max_derive_capteur * np.linspace(0.0, 1.0, end - s)
            affected += end - s
        return out, affected

    def injecter_donnees_manquantes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simule des défaillances de capteurs IoT réalistes plutôt qu'un MCAR uniforme :
        rafales de paquets perdus, capteurs figés, dérive de calibration.
        Généré indépendamment par (machine, capteur).
        """
        c = self.config
        df_copy = df.copy()
        colonnes_capteurs = [col for col in COLONNES_CAPTEURS if col in df_copy.columns]
        if not colonnes_capteurs or "id_machine" not in df_copy.columns or df_copy.empty:
            return df_copy

        donnees_capteurs = df_copy[colonnes_capteurs].to_numpy(dtype=float, copy=True)

        n_nan = n_stuck = n_drift = 0
        for positions in df_copy.groupby("id_machine", sort=False).indices.values():
            positions = np.asarray(positions)
            for col in range(len(colonnes_capteurs)):
                values = donnees_capteurs[positions, col]

                values, affected = self._appliquer_derive(values, float(np.nanstd(values)) or 1.0)
                n_drift += affected

                values, affected = self._appliquer_rafales(
                    values,
                    c.probabilite_debut_capteur_bloque,
                    c.pas_moyens_duree_capteur_bloque,
                    c.sigma_duree_capteur_bloque,
                    c.plage_duree_capteur_bloque,
                    mode="stuck",
                )
                n_stuck += affected

                values, affected = self._appliquer_rafales(
                    values,
                    c.probabilite_debut_rafale_nan_capteur,
                    c.pas_moyens_duree_rafale_nan_capteur,
                    c.sigma_duree_rafale_nan_capteur,
                    c.plage_duree_rafale_nan_capteur,
                    mode="nan",
                )
                n_nan += affected

                donnees_capteurs[positions, col] = values

        df_copy[colonnes_capteurs] = donnees_capteurs
        logger.info(
            f"Défaillance capteurs : {n_nan} valeurs manquantes (rafales), {n_stuck} figées, {n_drift} en dérive de calibration."
        )
        return df_copy

    def inserer_duckdb(self, df: pd.DataFrame, chemin_db: pathlib.Path):
        chemin_db.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(chemin_db)) as conn:
            conn.register("df_telemetrie", df)
            conn.execute(f"CREATE OR REPLACE TABLE {TABLE_TELEMETRIE} AS SELECT * FROM df_telemetrie")
            conn.execute(f"CREATE INDEX idx_machine_time ON {TABLE_TELEMETRIE}(id_machine, timestamp)")
        logger.info(f"Table {TABLE_TELEMETRIE} écrite dans {chemin_db} ({len(df):,} lignes).")
