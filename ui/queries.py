from datetime import date

import duckdb
import pandas as pd
import streamlit as st

from ui.constants import _PARTS, _PROD_COEFF, _QUAL_FIX, _TECH_RATE, DB_PATH, TABLE


@st.cache_data(ttl=300)
def q_kpis_period(date_from: str, date_to: str) -> dict:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        g = conn.execute(
            f"""SELECT (1 - AVG(target)) * 100 AS avail,
                       SUM(target)               AS failures,
                       SUM(power_loss_indicator) AS blackouts
                FROM {TABLE}
                WHERE timestamp BETWEEN ? AND ?""",
            [date_from, date_to],
        ).fetchone()
        mtbf_raw = conn.execute(
            f"""WITH m AS (
                    SELECT machine_id,
                           DATEDIFF('hour', MIN(timestamp), MAX(timestamp)) * 1.0
                           / NULLIF(SUM(target), 0) AS h
                    FROM {TABLE}
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY machine_id
                )
                SELECT AVG(h) FROM m""",
            [date_from, date_to],
        ).fetchone()[0]
        pm_windows = conn.execute(
            f"""SELECT COUNT(*) / 48
                FROM {TABLE}
                WHERE tool_wear BETWEEN 82 AND 90 AND target = 0
                  AND timestamp BETWEEN ? AND ?""",
            [date_from, date_to],
        ).fetchone()[0]
        blackouts_30d = conn.execute(
            f"""SELECT SUM(power_loss_indicator)
                FROM {TABLE}
                WHERE timestamp >= (SELECT MAX(timestamp) - INTERVAL 30 DAY FROM {TABLE})"""
        ).fetchone()[0]
    return dict(
        availability=float(g[0]),
        total_failures=int(g[1]),
        mtbf_h=float(mtbf_raw or 0.0),
        total_blackouts=int(g[2]),
        pm_windows=int(pm_windows),
        blackouts_30d=int(blackouts_30d),
    )


@st.cache_data(ttl=300)
def q_cost_period(date_from: str, date_to: str) -> dict:
    """
    C_panne = C_production + C_technique + C_qualite, par événement de panne.
    Les pannes consécutives (même machine, intervalles de 30 min) sont regroupées
    en un seul événement dont la durée = n_steps × 0.5 h.
    """
    Lv = _PROD_COEFF["L"] + _TECH_RATE
    Lf = _PARTS["L"] + _QUAL_FIX["L"]
    Mv = _PROD_COEFF["M"] + _TECH_RATE
    Mf = _PARTS["M"] + _QUAL_FIX["M"]
    Hv = _PROD_COEFF["H"] + _TECH_RATE
    Hf = _PARTS["H"] + _QUAL_FIX["H"]

    sql = f"""
    WITH failure_rows AS (
        SELECT machine_id, machine_type, timestamp,
               LAG(timestamp) OVER (PARTITION BY machine_id ORDER BY timestamp) AS prev_ts
        FROM {TABLE}
        WHERE target = 1 AND timestamp BETWEEN ? AND ?
    ),
    events AS (
        SELECT machine_id, machine_type,
               SUM(CASE WHEN prev_ts IS NULL
                        OR DATEDIFF('minute', prev_ts, timestamp) > 30
                        THEN 1 ELSE 0 END)
                   OVER (PARTITION BY machine_id ORDER BY timestamp) AS event_id
        FROM failure_rows
    ),
    durations AS (
        SELECT machine_id, machine_type, event_id,
               COUNT(*) * 0.5 AS dur_h
        FROM events
        GROUP BY machine_id, machine_type, event_id
    )
    SELECT
        COUNT(*)                                          AS n_events,
        SUM(CASE machine_type
            WHEN 'L' THEN {Lv} * dur_h + {Lf}
            WHEN 'M' THEN {Mv} * dur_h + {Mf}
            WHEN 'H' THEN {Hv} * dur_h + {Hf}
            ELSE 0 END)                                  AS total_cost,
        SUM(CASE machine_type
            WHEN 'L' THEN {_PROD_COEFF["L"]} * dur_h
            WHEN 'M' THEN {_PROD_COEFF["M"]} * dur_h
            WHEN 'H' THEN {_PROD_COEFF["H"]} * dur_h
            ELSE 0 END)                                  AS c_production,
        SUM(CASE machine_type
            WHEN 'L' THEN {_PARTS["L"]} + {_TECH_RATE} * dur_h
            WHEN 'M' THEN {_PARTS["M"]} + {_TECH_RATE} * dur_h
            WHEN 'H' THEN {_PARTS["H"]} + {_TECH_RATE} * dur_h
            ELSE 0 END)                                  AS c_technique,
        SUM(CASE machine_type
            WHEN 'L' THEN {_QUAL_FIX["L"]}
            WHEN 'M' THEN {_QUAL_FIX["M"]}
            WHEN 'H' THEN {_QUAL_FIX["H"]}
            ELSE 0 END)                                  AS c_qualite
    FROM durations
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        row = conn.execute(sql, [date_from, date_to]).fetchone()
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
def q_healthy_machines_pct(date_from: str, date_to: str) -> float:
    """% de machines sans aucune panne sur la période."""
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        result = conn.execute(
            f"""WITH machine_status AS (
                    SELECT machine_id, SUM(target) AS n_fail
                    FROM {TABLE}
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY machine_id
                )
                SELECT COUNT(*) FILTER (WHERE n_fail = 0) * 100.0 / COUNT(*)
                FROM machine_status""",
            [date_from, date_to],
        ).fetchone()[0]
    return float(result or 0.0)


@st.cache_data(ttl=300)
def q_failures_by_type_cause_period(date_from: str, date_to: str) -> "pd.DataFrame":
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return conn.execute(
            f"""SELECT machine_type,
                       failure_mode AS cause,
                       COUNT(*) AS n
                FROM {TABLE}
                WHERE target = 1 AND timestamp BETWEEN ? AND ?
                GROUP BY 1, 2
                ORDER BY 1, 2""",
            [date_from, date_to],
        ).fetchdf()


@st.cache_data(ttl=300)
def q_machine_ts_full(machine_id: str, steps: int) -> "pd.DataFrame":
    """Retourne la série temporelle complète avec toutes les features ML + indicateurs visuels."""
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        df = conn.execute(
            f"""SELECT timestamp,
                       process_temperature, ambient_temperature, tool_wear,
                       vibration, voltage_stability, target,
                       benin_season, machine_type, rotational_speed, torque,
                       activity_level, humidity, dust_concentration,
                       rain_flag, power_loss_indicator, voltage_level
                FROM {TABLE}
                WHERE machine_id = ?
                ORDER BY timestamp DESC LIMIT ?""",
            [machine_id, steps],
        ).fetchdf()
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=300)
def q_machine_list() -> list:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        rows = conn.execute(f"SELECT DISTINCT machine_id FROM {TABLE} ORDER BY machine_id").fetchdf()
    return list(rows["machine_id"])


@st.cache_data(ttl=300)
def q_machine_last(machine_id: str) -> dict:
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        row = conn.execute(
            f"""SELECT tool_wear, process_temperature, ambient_temperature,
                       vibration, activity_level, voltage_stability,
                       power_loss_indicator, machine_type,
                       dust_concentration, humidity, voltage_level,
                       rotational_speed, torque, rain_flag, benin_season
                FROM {TABLE}
                WHERE machine_id = ?
                  AND timestamp = (SELECT MAX(timestamp) FROM {TABLE} WHERE machine_id = ?)""",
            [machine_id, machine_id],
        ).fetchone()
        n_fail = conn.execute(f"SELECT SUM(target) FROM {TABLE} WHERE machine_id = ?", [machine_id]).fetchone()[0]
    keys = [
        "tool_wear",
        "process_temperature",
        "ambient_temperature",
        "vibration",
        "activity_level",
        "voltage_stability",
        "power_loss_indicator",
        "machine_type",
        "dust_concentration",
        "humidity",
        "voltage_level",
        "rotational_speed",
        "torque",
        "rain_flag",
        "benin_season",
    ]
    snap = dict(zip(keys, row))
    snap["n_failures"] = int(n_fail)
    return snap


@st.cache_data(ttl=300)
def q_top5_machines(date_from: str, date_to: str) -> "pd.DataFrame":
    """Top 5 machines par nombre de pannes sur la période, avec type et coût estimé."""
    Lv = _PROD_COEFF["L"] + _TECH_RATE
    Lf = _PARTS["L"] + _QUAL_FIX["L"]
    Mv = _PROD_COEFF["M"] + _TECH_RATE
    Mf = _PARTS["M"] + _QUAL_FIX["M"]
    Hv = _PROD_COEFF["H"] + _TECH_RATE
    Hf = _PARTS["H"] + _QUAL_FIX["H"]

    sql = f"""
    WITH failure_rows AS (
        SELECT machine_id, machine_type, timestamp,
               LAG(timestamp) OVER (PARTITION BY machine_id ORDER BY timestamp) AS prev_ts
        FROM {TABLE}
        WHERE target = 1 AND timestamp BETWEEN ? AND ?
    ),
    events AS (
        SELECT machine_id, machine_type,
               SUM(CASE WHEN prev_ts IS NULL OR DATEDIFF('minute', prev_ts, timestamp) > 30
                        THEN 1 ELSE 0 END)
                   OVER (PARTITION BY machine_id ORDER BY timestamp) AS event_id
        FROM failure_rows
    ),
    durations AS (
        SELECT machine_id, machine_type, event_id,
               COUNT(*) * 0.5 AS dur_h
        FROM events
        GROUP BY machine_id, machine_type, event_id
    )
    SELECT machine_id,
           machine_type,
           COUNT(*)  AS n_events,
           SUM(CASE machine_type
               WHEN 'L' THEN {Lv} * dur_h + {Lf}
               WHEN 'M' THEN {Mv} * dur_h + {Mf}
               WHEN 'H' THEN {Hv} * dur_h + {Hf}
               ELSE 0 END) AS total_cost
    FROM durations
    GROUP BY machine_id, machine_type
    ORDER BY n_events DESC
    LIMIT 5
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        df = conn.execute(sql, [date_from, date_to]).fetchdf()
    df["label"] = df["machine_id"].str.extract(r"(M_\d+)$")[0].fillna(df["machine_id"])
    return df


@st.cache_data(ttl=300)
def q_cost_timeline(date_from: str, date_to: str) -> "pd.DataFrame":
    """Coût des pannes agrégé par semaine ou mois selon l'étendue de la période."""
    days = (date.fromisoformat(date_to) - date.fromisoformat(date_from)).days
    trunc = "week" if days <= 90 else "month"

    Lv = _PROD_COEFF["L"] + _TECH_RATE
    Lf = _PARTS["L"] + _QUAL_FIX["L"]
    Mv = _PROD_COEFF["M"] + _TECH_RATE
    Mf = _PARTS["M"] + _QUAL_FIX["M"]
    Hv = _PROD_COEFF["H"] + _TECH_RATE
    Hf = _PARTS["H"] + _QUAL_FIX["H"]

    sql = f"""
    WITH failure_rows AS (
        SELECT machine_id, machine_type, timestamp,
               LAG(timestamp) OVER (PARTITION BY machine_id ORDER BY timestamp) AS prev_ts
        FROM {TABLE}
        WHERE target = 1 AND timestamp BETWEEN ? AND ?
    ),
    events AS (
        SELECT machine_id, machine_type, timestamp,
               SUM(CASE WHEN prev_ts IS NULL OR DATEDIFF('minute', prev_ts, timestamp) > 30
                        THEN 1 ELSE 0 END)
                   OVER (PARTITION BY machine_id ORDER BY timestamp) AS event_id
        FROM failure_rows
    ),
    event_agg AS (
        SELECT machine_id, machine_type, event_id,
               MIN(timestamp) AS event_start,
               COUNT(*) * 0.5 AS dur_h
        FROM events
        GROUP BY machine_id, machine_type, event_id
    )
    SELECT DATE_TRUNC('{trunc}', event_start) AS period,
           SUM(CASE machine_type
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
        df = conn.execute(sql, [date_from, date_to]).fetchdf()
    if not df.empty:
        df["cumulative_cost"] = df["cost"].cumsum()
    return df


@st.cache_resource
def get_predictor():
    from src.xpredict import Predictor

    return Predictor(use_mlflow=True)
