"""
Météo extérieure réelle de Glo-Djigbé via Open-Meteo : archive ERA5 (température, humidité, pluie) et qualité de
l'air CAMS (poussière d'Harmattan). Téléchargée une fois, puis gardée en cache dans DuckDB.
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

LATITUDE, LONGITUDE, FUSEAU = 6.62, 2.23, "Africa/Porto-Novo"
TABLE_METEO = "meteo_exterieure"
NOMS = {
    "temperature_2m": "temperature_exterieure",
    "relative_humidity_2m": "humidite_exterieure",
    "precipitation": "precipitation_mm_h",
    "dust": "poussiere_exterieure",
}


def _telecharger(url: str, variables: list[str], debut: date, fin: date) -> pd.DataFrame:
    parametres = {"latitude": LATITUDE, "longitude": LONGITUDE, "start_date": debut.isoformat(), "end_date": fin.isoformat(),
                  "hourly": ",".join(variables), "timezone": FUSEAU}  # fmt: skip
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(parametres)}", timeout=60) as reponse:
        df = pd.DataFrame(json.load(reponse)["hourly"])
    return df.assign(time=pd.to_datetime(df["time"])).set_index("time")


def telecharger_meteo(debut: date, fin: date) -> pd.DataFrame:
    """Météo horaire sur [debut, fin], heure locale béninoise (UTC+1, sans heure d'été)."""
    logger.info(f"Téléchargement Open-Meteo ({debut} → {fin})…")
    meteo = _telecharger("https://archive-api.open-meteo.com/v1/archive", list(NOMS)[:3], debut, fin)
    poussiere = _telecharger("https://air-quality-api.open-meteo.com/v1/air-quality", ["dust"], debut, fin)
    # Trous ponctuels possibles côté CAMS : interpolation dans le temps
    return meteo.join(poussiere, how="left").rename(columns=NOMS).interpolate(method="time", limit_direction="both")


def charger_meteo(chemin_db: pathlib.Path, debut: date, fin: date) -> pd.DataFrame:
    """Depuis le cache DuckDB s'il couvre la période, sinon téléchargée puis mise en cache."""
    chemin_db.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(chemin_db)) as conn:
        try:
            cache = conn.execute(
                f"SELECT * FROM {TABLE_METEO} WHERE time >= ? AND time < ? ORDER BY time", [debut, fin + timedelta(days=1)]
            ).fetchdf()
            if len(cache) == ((fin - debut).days + 1) * 24:
                return cache.set_index("time")
        except duckdb.CatalogException:
            pass
        df = telecharger_meteo(debut, fin)
        conn.register("df_meteo", df.reset_index())
        conn.execute(f"CREATE OR REPLACE TABLE {TABLE_METEO} AS SELECT * FROM df_meteo")
    return df


def au_pas(meteo_horaire: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Météo au pas de la simulation : interpolation des grandeurs continues, pluie de l'heure répartie sur ses sous-pas."""
    continues = meteo_horaire.drop(columns="precipitation_mm_h")
    df = continues.reindex(continues.index.union(timestamps)).interpolate(method="time").ffill().bfill().reindex(timestamps)
    sous_pas_par_heure = pd.Timedelta("1h") / pd.Series(timestamps).diff().median()
    cumul_heure = meteo_horaire["precipitation_mm_h"].reindex(timestamps.ceil("h")).fillna(0.0).to_numpy()
    df["precipitation_mm"] = cumul_heure / sous_pas_par_heure
    return df
