"""
Météo extérieure réelle de la Zone Industrielle de Glo-Djigbé, via Open-Meteo :
- archive ERA5 : température, humidité relative, précipitations
- qualité de l'air CAMS : poussière désertique (Harmattan)

Téléchargée une seule fois puis mise en cache dans DuckDB (table meteo_exterieure).
"""

import json
import logging
import pathlib
import urllib.parse
import urllib.request
from datetime import date, timedelta

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

LATITUDE, LONGITUDE = 6.62, 2.23
FUSEAU = "Africa/Porto-Novo"
TABLE_METEO = "meteo_exterieure"

_URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_URL_QUALITE_AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"
_VARIABLES = {
    "temperature_2m": "temperature_exterieure",
    "relative_humidity_2m": "humidite_exterieure",
    "precipitation": "precipitation_mm_h",
    "dust": "poussiere_exterieure",
}


def _telecharger(url: str, variables: list[str], debut: date, fin: date) -> pd.DataFrame:
    parametres = urllib.parse.urlencode(
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": debut.isoformat(),
            "end_date": fin.isoformat(),
            "hourly": ",".join(variables),
            "timezone": FUSEAU,
        }
    )
    with urllib.request.urlopen(f"{url}?{parametres}", timeout=60) as reponse:
        horaire = json.load(reponse)["hourly"]
    df = pd.DataFrame(horaire)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def telecharger_meteo(debut: date, fin: date) -> pd.DataFrame:
    """Météo horaire extérieure sur [debut, fin], heure locale béninoise (UTC+1, sans heure d'été)."""
    logger.info(f"Téléchargement Open-Meteo ({debut} → {fin}, lat={LATITUDE}, lon={LONGITUDE})...")
    meteo = _telecharger(_URL_ARCHIVE, ["temperature_2m", "relative_humidity_2m", "precipitation"], debut, fin)
    poussiere = _telecharger(_URL_QUALITE_AIR, ["dust"], debut, fin)
    df = meteo.join(poussiere, how="left").rename(columns=_VARIABLES)
    # Trous ponctuels possibles côté CAMS : interpolation temporelle, jamais de valeur inventée en bord
    return df.interpolate(method="time", limit_direction="both")


def charger_meteo(chemin_db: pathlib.Path, debut: date, fin: date) -> pd.DataFrame:
    """Météo horaire depuis le cache DuckDB si elle couvre la période, sinon téléchargée puis mise en cache."""
    heures_attendues = ((fin - debut).days + 1) * 24
    chemin_db.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(chemin_db)) as conn:
        try:
            cache = conn.execute(
                f"SELECT * FROM {TABLE_METEO} WHERE time >= ? AND time < ? ORDER BY time",
                [debut, fin + timedelta(days=1)],
            ).fetchdf()
            if len(cache) == heures_attendues:
                logger.info(f"Météo chargée depuis le cache DuckDB ({len(cache):,} heures).")
                return cache.set_index("time")
        except duckdb.CatalogException:
            pass

        df = telecharger_meteo(debut, fin)
        conn.register("df_meteo", df.reset_index())
        conn.execute(f"CREATE OR REPLACE TABLE {TABLE_METEO} AS SELECT * FROM df_meteo")
        logger.info(f"Météo mise en cache dans {chemin_db} ({len(df):,} heures).")
    return df


def au_pas(meteo_horaire: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Ramène la météo horaire au pas de la simulation : interpolation linéaire pour les grandeurs
    continues, et répartition égale des précipitations (cumul de l'heure écoulée) sur ses sous-pas.
    """
    continues = meteo_horaire.drop(columns="precipitation_mm_h")
    df = continues.reindex(continues.index.union(timestamps)).interpolate(method="time").ffill().bfill().reindex(timestamps)

    sous_pas_par_heure = pd.Timedelta("1h") / pd.Series(timestamps).diff().median()
    cumul_heure = meteo_horaire["precipitation_mm_h"].reindex(timestamps.ceil("h")).fillna(0.0).to_numpy()
    df["precipitation_mm"] = cumul_heure / sous_pas_par_heure
    return df
