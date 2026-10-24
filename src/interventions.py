"""
Interventions : chaque alerte du modèle devient une intervention à traiter, planifiée, puis qualifiée par un constat.
Base DuckDB séparée, écrite par l'API seule. L'historique 2025 est reconstitué à partir des alertes et des pannes réelles :
une alerte suivie d'une panne prévisible dans l'horizon est fondée, les autres sont de fausses alertes.
Usage : python -m src.interventions (après make score, API arrêtée)
"""

import logging
import pathlib

import duckdb
import numpy as np
import pandas as pd

from src.config import CHEMIN_DB, CHEMIN_INTERVENTIONS, PAS, TABLE, TABLE_PREDICTIONS
from src.donnees import DATE_TEST, MODES_PANNE_PREDICTIBLES
from src.evaluation import PAS_HORIZON, _episodes_alerte
from src.usine import TECHNICIENS, technicien_conseille

logger = logging.getLogger(__name__)

STATUTS = ("a_traiter", "planifiee", "qualifiee")
CONSTATS = {"corrige": True, "panne": True, "rien": False, "ecartee": False}  # constat -> alerte fondée ?
TYPES = ("Inspection", "Entretien", "Réparation")
SPECIALITE_PAR_CAUSE = {"HDF": "Électrique"}  # surchauffe : moteur et armoire ; le reste est mécanique
ECART_FUSION = 48  # pas de 30 min : deux alertes à moins de 24 h d'écart forment une seule intervention

_SCHEMA = """
    CREATE TABLE interventions (
        id INTEGER PRIMARY KEY,
        id_machine VARCHAR NOT NULL,
        debut_alerte TIMESTAMP NOT NULL,
        statut VARCHAR NOT NULL,
        technicien VARCHAR,
        creneau VARCHAR,
        type VARCHAR,
        constat VARCHAR,
        fondee BOOLEAN,
        source VARCHAR NOT NULL,  -- historique (reconstitué) ou terrain (saisi dans le tableau de bord)
        maj TIMESTAMP
    )
"""


def _fusionner(episodes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    fusion: list[tuple[int, int]] = []
    for debut, fin in episodes:
        if fusion and debut - fusion[-1][1] <= ECART_FUSION:
            fusion[-1] = (fusion[-1][0], fin)
        else:
            fusion.append((debut, fin))
    return fusion


def episodes_alerte(df: pd.DataFrame) -> pd.DataFrame:
    """df : id_machine, timestamp, panne, mode_panne, niveau, trié. Une ligne par épisode d'alerte, fondé ou non."""
    lignes = []
    for id_machine, g in df.groupby("id_machine", sort=False):
        g = g.reset_index(drop=True)
        alertes = (g["niveau"] != "normal").to_numpy()
        est_panne = ((g["panne"] == 1) & g["mode_panne"].isin(MODES_PANNE_PREDICTIBLES)).to_numpy()
        pannes = np.flatnonzero(est_panne)
        for debut, fin in _fusionner(_episodes_alerte(alertes, pas_max_episode=None)):
            suivantes = pannes[(pannes >= debut) & (pannes - fin <= PAS_HORIZON)]
            lignes.append(
                {
                    "id_machine": id_machine,
                    "debut_alerte": g.loc[debut, "timestamp"],
                    "en_cours": fin == len(g) - 1,
                    "fondee": len(suivantes) > 0,
                    "cause": g.loc[suivantes[0], "mode_panne"] if len(suivantes) else None,
                }
            )
    return pd.DataFrame(lignes)


def initialiser(chemin: pathlib.Path = CHEMIN_INTERVENTIONS) -> int:
    """(Re)crée la base : historique 2025 qualifié, et une intervention à traiter par alerte en cours."""
    with duckdb.connect(str(CHEMIN_DB), read_only=True) as conn:
        df = conn.execute(
            f"""SELECT t.id_machine, t.timestamp, t.panne, t.mode_panne, t.etat, p.niveau
                FROM {TABLE} t JOIN {TABLE_PREDICTIONS} p USING (id_machine, timestamp)
                WHERE t.timestamp >= ? ORDER BY t.id_machine, t.timestamp""",
            [DATE_TEST],
        ).fetchdf()
    ep = episodes_alerte(df)
    en_panne = set(df.groupby("id_machine").tail(1).query("etat == 'panne'")["id_machine"])
    ep = ep[~(ep["en_cours"] & ep["id_machine"].isin(en_panne))]

    historique = ep[~ep["en_cours"]]
    lignes = pd.DataFrame(
        {
            "id_machine": ep["id_machine"],
            "debut_alerte": ep["debut_alerte"],
            "statut": np.where(ep["en_cours"], "a_traiter", "qualifiee"),
            "technicien": [
                None if en_cours else technicien_conseille(SPECIALITE_PAR_CAUSE.get(cause, "Mécanique"), debut.hour)
                for en_cours, cause, debut in zip(ep["en_cours"], ep["cause"], ep["debut_alerte"])
            ],
            "creneau": [None if c else (d + 4 * PAS).strftime("%d/%m %H:%M") for c, d in zip(ep["en_cours"], ep["debut_alerte"])],
            "type": [None if c else "Inspection" for c in ep["en_cours"]],
            "constat": [None if c else ("panne" if f else "rien") for c, f in zip(ep["en_cours"], ep["fondee"])],
            "fondee": [None if c else bool(f) for c, f in zip(ep["en_cours"], ep["fondee"])],
            "source": "historique",
            "maj": ep["debut_alerte"],
        }
    ).sort_values("debut_alerte")
    lignes.insert(0, "id", range(1, len(lignes) + 1))

    chemin.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(chemin)) as conn:
        conn.execute("DROP TABLE IF EXISTS interventions")
        conn.execute(_SCHEMA)
        conn.register("df_interventions", lignes)
        conn.execute("INSERT INTO interventions SELECT * FROM df_interventions")
    fondees, a_traiter = int(historique["fondee"].sum()), int(ep["en_cours"].sum())
    logger.info(f"{len(historique):,} interventions historiques ({fondees} fondées), {a_traiter} à traiter.")
    return len(lignes)


def connexion(chemin: pathlib.Path = CHEMIN_INTERVENTIONS) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(chemin))


def lister(conn: duckdb.DuckDBPyConnection, statut: str) -> pd.DataFrame:
    return conn.execute("SELECT * FROM interventions WHERE statut = ? ORDER BY debut_alerte DESC", [statut]).fetchdf()


def historique_recent(conn: duckdb.DuckDBPyConnection, n: int = 8) -> pd.DataFrame:
    return conn.execute(
        "SELECT * FROM interventions WHERE statut = 'qualifiee' ORDER BY maj DESC, debut_alerte DESC LIMIT ?", [n]
    ).fetchdf()


def fiabilite(conn: duckdb.DuckDBPyConnection) -> dict:
    """Sur toutes les alertes qualifiées : combien ont été suivies d'une panne (ou d'un défaut trouvé)."""
    total, fondees = conn.execute(
        "SELECT COUNT(*), COUNT(*) FILTER (WHERE fondee) FROM interventions WHERE statut = 'qualifiee'"
    ).fetchone() or (0, 0)
    return {"alertes_qualifiees": total, "alertes_fondees": fondees, "part_fondees": fondees / total if total else None}


def charge_techniciens(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    charges = dict(
        conn.execute("SELECT technicien, COUNT(*) FROM interventions WHERE statut = 'planifiee' GROUP BY 1").fetchall()
    )
    return [{**t, "interventions_planifiees": charges.get(t["id"], 0)} for t in TECHNICIENS]


def _modifier(conn: duckdb.DuckDBPyConnection, id_intervention: int, statuts_permis: tuple, champs: dict) -> bool:
    statut = conn.execute("SELECT statut FROM interventions WHERE id = ?", [id_intervention]).fetchone()
    if statut is None or statut[0] not in statuts_permis:
        return False
    affectations = ", ".join(f"{k} = ?" for k in champs)
    conn.execute(
        f"UPDATE interventions SET {affectations}, source = 'terrain', maj = now() WHERE id = ?",
        [*champs.values(), id_intervention],
    )
    return True


def planifier(conn, id_intervention: int, technicien: str, creneau: str, type_intervention: str) -> bool:
    if technicien not in {t["id"] for t in TECHNICIENS} or type_intervention not in TYPES:
        raise ValueError("Technicien ou type d'intervention inconnu.")
    champs = {"statut": "planifiee", "technicien": technicien, "creneau": creneau, "type": type_intervention}
    return _modifier(conn, id_intervention, ("a_traiter",), champs)


def qualifier(conn, id_intervention: int, constat: str) -> bool:
    """Constat après intervention, ou alerte écartée sans intervention."""
    if constat not in CONSTATS:
        raise ValueError(f"Constat inconnu : {constat}")
    statuts = ("a_traiter",) if constat == "ecartee" else ("planifiee",)
    return _modifier(conn, id_intervention, statuts, {"statut": "qualifiee", "constat": constat, "fondee": CONSTATS[constat]})


def rouvrir(conn, id_intervention: int) -> bool:
    """Annule un constat saisi par erreur : retour à planifiée si un technicien était assigné, sinon à traiter."""
    ligne = conn.execute(
        "SELECT technicien, source FROM interventions WHERE id = ? AND statut = 'qualifiee'", [id_intervention]
    ).fetchone()
    if ligne is None or ligne[1] != "terrain":
        return False
    statut = "planifiee" if ligne[0] else "a_traiter"
    return _modifier(conn, id_intervention, ("qualifiee",), {"statut": statut, "constat": None, "fondee": None})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    initialiser()
