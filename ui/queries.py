from datetime import date

import duckdb
import pandas as pd
import streamlit as st

from ui.constants import _PARTS, _PROD_COEFF, _QUAL_FIX, _TECH_RATE, DB_PATH, TABLE

_SQL_EVENEMENTS_COUPURE = """
    WITH etat_reseau AS (
        SELECT DISTINCT timestamp, indicateur_coupure
        FROM {table}
        WHERE {filtre}
    ),
    transitions AS (
        SELECT indicateur_coupure - LAG(indicateur_coupure, 1, 0) OVER (ORDER BY timestamp) AS debut
        FROM etat_reseau
    )
    SELECT COUNT(*) FROM transitions WHERE debut = 1
"""


@st.cache_data(ttl=300)
def q_kpis_periode(date_debut: str, date_fin: str) -> dict:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        g = conn.execute(
            f"""SELECT (1 - AVG(panne)) * 100 AS avail, SUM(panne) AS failures
                FROM {TABLE}
                WHERE timestamp BETWEEN ? AND ?""",
            [date_debut, date_fin],
        ).fetchone()
        mtbf_raw = conn.execute(
            f"""WITH m AS (
                    SELECT id_machine, DATEDIFF('hour', MIN(timestamp), MAX(timestamp)) * 1.0 / NULLIF(SUM(panne), 0) AS h
                    FROM {TABLE}
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY id_machine
                )
                SELECT AVG(h) FROM m""",
            [date_debut, date_fin],
        ).fetchone()[0]
        pm_windows = conn.execute(
            f"""SELECT COUNT(*) / 48
                FROM {TABLE}
                WHERE usure_outil BETWEEN 82 AND 90 AND panne = 0
                  AND timestamp BETWEEN ? AND ?""",
            [date_debut, date_fin],
        ).fetchone()[0]
        total_blackouts = conn.execute(
            _SQL_EVENEMENTS_COUPURE.format(table=TABLE, filtre="timestamp BETWEEN ? AND ?"),
            [date_debut, date_fin],
        ).fetchone()[0]
        blackouts_30d = conn.execute(
            _SQL_EVENEMENTS_COUPURE.format(
                table=TABLE, filtre=f"timestamp >= (SELECT MAX(timestamp) - INTERVAL 30 DAY FROM {TABLE})"
            )
        ).fetchone()[0]
    return dict(
        availability=float(g[0]),
        total_pannes=int(g[1]),
        mtbf_h=float(mtbf_raw or 0.0),
        total_blackouts=int(total_blackouts),
        pm_windows=int(pm_windows),
        blackouts_30d=int(blackouts_30d),
    )


@st.cache_data(ttl=300)
def q_cout_periode(date_debut: str, date_fin: str) -> dict:
    """
    C_panne = C_production + C_technique + C_qualite, par événement de panne.
    Les pannes consécutives (même machine, intervalles de 30 min) sont regroupées en 1 seul événement.
    """
    Lv = _PROD_COEFF["L"] + _TECH_RATE
    Lf = _PARTS["L"] + _QUAL_FIX["L"]
    Mv = _PROD_COEFF["M"] + _TECH_RATE
    Mf = _PARTS["M"] + _QUAL_FIX["M"]
    Hv = _PROD_COEFF["H"] + _TECH_RATE
    Hf = _PARTS["H"] + _QUAL_FIX["H"]

    sql = f"""
    WITH failure_rows AS (
        SELECT id_machine, type_machine, timestamp,
               LAG(timestamp) OVER (PARTITION BY id_machine ORDER BY timestamp) AS prev_ts
        FROM {TABLE}
        WHERE panne = 1 AND timestamp BETWEEN ? AND ?
    ),
    events AS (
        SELECT id_machine, type_machine,
               SUM(CASE WHEN prev_ts IS NULL
                        OR DATEDIFF('minute', prev_ts, timestamp) > 30
                        THEN 1 ELSE 0 END)
                   OVER (PARTITION BY id_machine ORDER BY timestamp) AS event_id
        FROM failure_rows
    ),
    durations AS (
        SELECT id_machine, type_machine, event_id,
               COUNT(*) * 0.5 AS dur_h
        FROM events
        GROUP BY id_machine, type_machine, event_id
    )
    SELECT
        COUNT(*)                                          AS n_events,
        SUM(CASE type_machine
            WHEN 'L' THEN {Lv} * dur_h + {Lf}
            WHEN 'M' THEN {Mv} * dur_h + {Mf}
            WHEN 'H' THEN {Hv} * dur_h + {Hf}
            ELSE 0 END)                                  AS total_cost,
        SUM(CASE type_machine
            WHEN 'L' THEN {_PROD_COEFF["L"]} * dur_h
            WHEN 'M' THEN {_PROD_COEFF["M"]} * dur_h
            WHEN 'H' THEN {_PROD_COEFF["H"]} * dur_h
            ELSE 0 END)                                  AS c_production,
        SUM(CASE type_machine
            WHEN 'L' THEN {_PARTS["L"]} + {_TECH_RATE} * dur_h
            WHEN 'M' THEN {_PARTS["M"]} + {_TECH_RATE} * dur_h
            WHEN 'H' THEN {_PARTS["H"]} + {_TECH_RATE} * dur_h
            ELSE 0 END)                                  AS c_technique,
        SUM(CASE type_machine
            WHEN 'L' THEN {_QUAL_FIX["L"]}
            WHEN 'M' THEN {_QUAL_FIX["M"]}
            WHEN 'H' THEN {_QUAL_FIX["H"]}
            ELSE 0 END)                                  AS c_qualite
    FROM durations
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        row = conn.execute(sql, [date_debut, date_fin]).fetchone()
    if not row or not row[0]:
        return {"n_events": 0, "total_cost": 0.0, "c_production": 0.0, "c_technique": 0.0, "c_qualite": 0.0}
    return {
        "n_events": int(row[0]),
        "total_cost": float(row[1] or 0),
        "c_production": float(row[2] or 0),
        "c_technique": float(row[3] or 0),
        "c_qualite": float(row[4] or 0),
    }


@st.cache_data(ttl=300)
def q_pourcentage_machines_saines(date_debut: str, date_fin: str) -> float:
    """% de machines sans aucune panne sur la période."""
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        result = conn.execute(
            f"""WITH machine_status AS (
                    SELECT id_machine, SUM(panne) AS n_fail
                    FROM {TABLE}
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY id_machine
                )
                SELECT COUNT(*) FILTER (WHERE n_fail = 0) * 100.0 / COUNT(*)
                FROM machine_status""",
            [date_debut, date_fin],
        ).fetchone()[0]
    return float(result or 0.0)


@st.cache_data(ttl=300)
def q_pannes_par_type_cause_periode(date_debut: str, date_fin: str) -> "pd.DataFrame":
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return conn.execute(
            f"""SELECT type_machine,
                       mode_panne AS cause,
                       COUNT(*) AS n
                FROM {TABLE}
                WHERE panne = 1 AND timestamp BETWEEN ? AND ?
                GROUP BY 1, 2
                ORDER BY 1, 2""",
            [date_debut, date_fin],
        ).fetchdf()


@st.cache_data(ttl=300)
def q_serie_complete_machine(id_machine: str, steps: int) -> "pd.DataFrame":
    """Retourne la série temporelle complète avec toutes les features ML + indicateurs visuels."""
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        df = conn.execute(
            f"""SELECT timestamp,
                       temperature_process, temperature_ambiante, usure_outil,
                       vibration, stabilite_tension, panne,
                       saison_benin, type_machine, vitesse_rotation, couple,
                       niveau_activite, humidite, concentration_poussiere,
                       indicateur_pluie, indicateur_coupure, niveau_tension,
                       encrassement
                FROM {TABLE}
                WHERE id_machine = ?
                ORDER BY timestamp DESC LIMIT ?""",
            [id_machine, steps],
        ).fetchdf()
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=300)
def q_liste_machines() -> list:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        rows = conn.execute(f"SELECT DISTINCT id_machine FROM {TABLE} ORDER BY id_machine").fetchdf()
    return list(rows["id_machine"])


@st.cache_data(ttl=300)
def q_dernier_releve_machine(id_machine: str) -> dict:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        row = conn.execute(
            f"""SELECT usure_outil, temperature_process, temperature_ambiante,
                       vibration, niveau_activite, stabilite_tension,
                       indicateur_coupure, type_machine,
                       concentration_poussiere, humidite, niveau_tension,
                       vitesse_rotation, couple, indicateur_pluie, saison_benin,
                       encrassement
                FROM {TABLE}
                WHERE id_machine = ?
                  AND timestamp = (SELECT MAX(timestamp) FROM {TABLE} WHERE id_machine = ?)""",
            [id_machine, id_machine],
        ).fetchone()
        n_fail = conn.execute(f"SELECT SUM(panne) FROM {TABLE} WHERE id_machine = ?", [id_machine]).fetchone()[0]
    keys = [
        "usure_outil",
        "temperature_process",
        "temperature_ambiante",
        "vibration",
        "niveau_activite",
        "stabilite_tension",
        "indicateur_coupure",
        "type_machine",
        "concentration_poussiere",
        "humidite",
        "niveau_tension",
        "vitesse_rotation",
        "couple",
        "indicateur_pluie",
        "saison_benin",
        "encrassement",
    ]
    snap = dict(zip(keys, row))
    snap["nb_pannes"] = int(n_fail)
    return snap


@st.cache_data(ttl=300)
def q_top5_machines(date_debut: str, date_fin: str) -> "pd.DataFrame":
    """Top 5 machines par nombre de pannes sur la période, avec type et coût estimé."""
    Lv = _PROD_COEFF["L"] + _TECH_RATE
    Lf = _PARTS["L"] + _QUAL_FIX["L"]
    Mv = _PROD_COEFF["M"] + _TECH_RATE
    Mf = _PARTS["M"] + _QUAL_FIX["M"]
    Hv = _PROD_COEFF["H"] + _TECH_RATE
    Hf = _PARTS["H"] + _QUAL_FIX["H"]

    sql = f"""
    WITH failure_rows AS (
        SELECT id_machine, type_machine, timestamp,
               LAG(timestamp) OVER (PARTITION BY id_machine ORDER BY timestamp) AS prev_ts
        FROM {TABLE}
        WHERE panne = 1 AND timestamp BETWEEN ? AND ?
    ),
    events AS (
        SELECT id_machine, type_machine,
               SUM(CASE WHEN prev_ts IS NULL OR DATEDIFF('minute', prev_ts, timestamp) > 30
                        THEN 1 ELSE 0 END)
                   OVER (PARTITION BY id_machine ORDER BY timestamp) AS event_id
        FROM failure_rows
    ),
    durations AS (
        SELECT id_machine, type_machine, event_id,
               COUNT(*) * 0.5 AS dur_h
        FROM events
        GROUP BY id_machine, type_machine, event_id
    )
    SELECT id_machine,
           type_machine,
           COUNT(*)  AS n_events,
           SUM(CASE type_machine
               WHEN 'L' THEN {Lv} * dur_h + {Lf}
               WHEN 'M' THEN {Mv} * dur_h + {Mf}
               WHEN 'H' THEN {Hv} * dur_h + {Hf}
               ELSE 0 END) AS total_cost
    FROM durations
    GROUP BY id_machine, type_machine
    ORDER BY n_events DESC
    LIMIT 5
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        df = conn.execute(sql, [date_debut, date_fin]).fetchdf()
    df["label"] = df["id_machine"].str.extract(r"(M_\d+)$")[0].fillna(df["id_machine"])
    return df


@st.cache_data(ttl=300)
def q_chronologie_cout(date_debut: str, date_fin: str) -> "pd.DataFrame":
    """Coût des pannes agrégé par semaine ou mois selon l'étendue de la période."""
    days = (date.fromisoformat(date_fin) - date.fromisoformat(date_debut)).days
    trunc = "week" if days <= 90 else "month"

    Lv = _PROD_COEFF["L"] + _TECH_RATE
    Lf = _PARTS["L"] + _QUAL_FIX["L"]
    Mv = _PROD_COEFF["M"] + _TECH_RATE
    Mf = _PARTS["M"] + _QUAL_FIX["M"]
    Hv = _PROD_COEFF["H"] + _TECH_RATE
    Hf = _PARTS["H"] + _QUAL_FIX["H"]

    sql = f"""
    WITH failure_rows AS (
        SELECT id_machine, type_machine, timestamp,
               LAG(timestamp) OVER (PARTITION BY id_machine ORDER BY timestamp) AS prev_ts
        FROM {TABLE}
        WHERE panne = 1 AND timestamp BETWEEN ? AND ?
    ),
    events AS (
        SELECT id_machine, type_machine, timestamp,
               SUM(CASE WHEN prev_ts IS NULL OR DATEDIFF('minute', prev_ts, timestamp) > 30
                        THEN 1 ELSE 0 END)
                   OVER (PARTITION BY id_machine ORDER BY timestamp) AS event_id
        FROM failure_rows
    ),
    event_agg AS (
        SELECT id_machine, type_machine, event_id,
               MIN(timestamp) AS event_start,
               COUNT(*) * 0.5 AS dur_h
        FROM events
        GROUP BY id_machine, type_machine, event_id
    )
    SELECT DATE_TRUNC('{trunc}', event_start) AS period,
           SUM(CASE type_machine
               WHEN 'L' THEN {Lv} * dur_h + {Lf}
               WHEN 'M' THEN {Mv} * dur_h + {Mf}
               WHEN 'H' THEN {Hv} * dur_h + {Hf}
               ELSE 0 END)                    AS cost,
           COUNT(*)                           AS n_events
    FROM event_agg
    GROUP BY period
    ORDER BY period
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        df = conn.execute(sql, [date_debut, date_fin]).fetchdf()
    if not df.empty:
        df["cumulative_cost"] = df["cost"].cumsum()
    return df


@st.cache_resource
def obtenir_predicteur():
    from src.xpredict import Predicteur

    return Predicteur(utiliser_mlflow=True)


@st.cache_resource
def obtenir_explicateur():
    """Explicateur SHAP (point 33) — construit une seule fois sur le pipeline du
    champion courant, réutilisé pour toutes les machines de la vue "à traiter"."""
    from src.explain import ExplicateurCause

    return ExplicateurCause(obtenir_predicteur().pipeline)


@st.cache_data(ttl=300)
def q_derniers_releves_toutes_machines(fenetre_historique: int) -> pd.DataFrame:
    """
    Les `fenetre_historique` dernières lignes de CHAQUE machine (pas juste la
    dernière) — nécessaire pour calculer les features glissantes (24h) avant de
    prédire, utilisé par la vue "à traiter aujourd'hui" (point 34).
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        df = conn.execute(
            f"""SELECT * FROM (
                    SELECT id_machine, type_machine, timestamp,
                           temperature_process, temperature_ambiante, usure_outil,
                           vibration, stabilite_tension, saison_benin, vitesse_rotation,
                           couple, niveau_activite, humidite, concentration_poussiere,
                           indicateur_pluie, indicateur_coupure, niveau_tension, encrassement,
                           ROW_NUMBER() OVER (PARTITION BY id_machine ORDER BY timestamp DESC) AS rang
                    FROM {TABLE}
                )
                WHERE rang <= ?
                ORDER BY id_machine, timestamp""",
            [fenetre_historique],
        ).fetchdf()
    return df.drop(columns="rang")


@st.cache_data(ttl=300)
def q_priorites_parc() -> pd.DataFrame:
    """
    Priorités du parc (point 34), mises en cache une seule fois — réutilisées à la
    fois par le tableau "à traiter aujourd'hui" (tab_general) et par la sélection par
    défaut de la Vue Machine (tab_machine), pour ne pas recalculer deux fois la même
    boucle prédiction + SHAP sur 50 machines à chaque chargement de page.
    """
    from src.features import FENETRE_24H
    from ui.priorites import calculer_priorites_parc

    predicteur = obtenir_predicteur()
    explicateur = obtenir_explicateur()
    df_historique = q_derniers_releves_toutes_machines(FENETRE_24H + 4)
    return calculer_priorites_parc(df_historique, predicteur, explicateur)
