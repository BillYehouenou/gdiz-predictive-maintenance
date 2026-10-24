"""
Calculs servis au tableau de bord : indicateurs par période, pannes, coûts, état du parc, fiche machine.
Lit la télémétrie et la table predictions dans DuckDB ; le front n'affiche que ce qui sort d'ici.
"""

from datetime import timedelta
from functools import lru_cache

import duckdb
import pandas as pd

from src.config import CHEMIN_DB, PAS, TABLE, TABLE_PREDICTIONS
from src.donnees import DATE_TEST
from src.usine import cout_panne, criticite, nom_machine, sante, statut, zone

PERIODES = {"30j": "30 j", "t4": "T4", "s2": "S2", "annee": "Année", "tout": "2024-2025"}
CAPTEURS_FICHE = {
    "vibration": ("Vibration max", "mm/s", "max"),
    "temperature_process": ("Temp. process max", "°C", "max"),
    "couple": ("Couple max", "Nm", "max"),
    "concentration_poussiere": ("Poussière max", "µg/m³", "max"),
    "niveau_tension": ("Tension min", "V", "min"),
}


def connexion() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(CHEMIN_DB), read_only=True)


def _ligne(conn: duckdb.DuckDBPyConnection, sql: str, parametres: list | None = None) -> tuple:
    ligne = conn.execute(sql, parametres or []).fetchone()
    assert ligne is not None
    return ligne


def fin_donnees(conn: duckdb.DuckDBPyConnection) -> pd.Timestamp:
    return pd.Timestamp(_ligne(conn, f"SELECT MAX(timestamp) FROM {TABLE}")[0])


def bornes_periode(code: str, fin: pd.Timestamp) -> dict:
    """Début et fin (exclue) de la période, et de la période précédente de même nature."""
    fin_excl = fin.normalize() + timedelta(days=1)
    annee = fin.year
    if code == "30j":
        debut = fin_excl - timedelta(days=30)
        prec = (debut - timedelta(days=30), debut)
        libelles = ("30 derniers jours", "30 jours précédents")
    elif code == "t4":
        debut = pd.Timestamp(annee, 10, 1)
        prec = (pd.Timestamp(annee, 7, 1), debut)
        libelles = (f"T4 {annee}", f"T3 {annee}")
    elif code == "s2":
        debut = pd.Timestamp(annee, 7, 1)
        prec = (pd.Timestamp(annee, 1, 1), debut)
        libelles = (f"S2 {annee}", f"S1 {annee}")
    elif code == "annee":
        debut = pd.Timestamp(annee, 1, 1)
        prec = (pd.Timestamp(annee - 1, 1, 1), debut)
        libelles = (f"Année {annee}", str(annee - 1))
    elif code == "tout":
        return {"debut": pd.Timestamp("2000-01-01"), "fin": fin_excl, "libelle": "2024-2025", "precedente": None}
    else:
        raise ValueError(f"Période inconnue : {code}")
    return {
        "debut": debut,
        "fin": fin_excl,
        "libelle": libelles[0],
        "precedente": {"debut": prec[0], "fin": prec[1], "libelle": libelles[1]},
    }


def evenements(conn: duckdb.DuckDBPyConnection, debut, fin, id_machine: str | None = None) -> pd.DataFrame:
    """Une ligne par panne : machine, gamme, date, cause, durée réelle de réparation (h), coût (FCFA)."""
    filtre = "AND id_machine = ?" if id_machine else ""
    df = conn.execute(
        f"""WITH t AS (
                SELECT id_machine, type_machine AS gamme, timestamp, panne, mode_panne, etat,
                       SUM(panne) OVER (PARTITION BY id_machine ORDER BY timestamp) AS n
                FROM {TABLE} WHERE timestamp >= ? AND timestamp < ? {filtre}
            )
            SELECT id_machine, ANY_VALUE(gamme) AS gamme,
                   MIN(timestamp) FILTER (WHERE panne = 1) AS date,
                   ANY_VALUE(mode_panne) FILTER (WHERE panne = 1) AS cause,
                   COUNT(*) FILTER (WHERE etat = 'panne' AND panne = 0) * {PAS / pd.Timedelta("1h")} AS duree_h
            FROM t WHERE n > 0 AND (panne = 1 OR etat = 'panne')
            GROUP BY id_machine, n
            HAVING date IS NOT NULL
            ORDER BY date""",
        [debut, fin] + ([id_machine] if id_machine else []),
    ).fetchdf()
    df["cout"] = [cout_panne(g, max(h, PAS / pd.Timedelta("1h"))) for g, h in zip(df["gamme"], df["duree_h"])]
    return df


def indicateurs(conn: duckdb.DuckDBPyConnection, debut, fin, id_machine: str | None = None) -> dict:
    """Disponibilité (hors arrêts planifiés et délestages), MTBF (heures de marche / pannes), coût, heures d'arrêt."""
    filtre = "AND id_machine = ?" if id_machine else ""
    marche, panne = _ligne(
        conn,
        f"""SELECT COUNT(*) FILTER (WHERE etat = 'marche'), COUNT(*) FILTER (WHERE etat = 'panne')
            FROM {TABLE} WHERE timestamp >= ? AND timestamp < ? {filtre}""",
        [debut, fin] + ([id_machine] if id_machine else []),
    )
    ev = evenements(conn, debut, fin, id_machine)
    heures_marche = marche * PAS / pd.Timedelta("1h")
    return {
        "disponibilite": 1 - panne / (marche + panne) if marche + panne else None,
        "mtbf_h": heures_marche / len(ev) if len(ev) else None,
        "cout": float(ev["cout"].sum()),
        "heures_arret": float(ev["duree_h"].sum()),
        "heures_marche": heures_marche,
        "pannes": len(ev),
    }


def tableau_parc(conn: duckdb.DuckDBPyConnection, code: str) -> dict:
    """Tout l'écran Parc pour une période : indicateurs (et période précédente), pannes par gamme et cause, coût."""
    b = bornes_periode(code, fin_donnees(conn))
    prec = b["precedente"]
    ev = evenements(conn, b["debut"], b["fin"])
    debut_effectif = max(b["debut"], pd.Timestamp(_ligne(conn, f"SELECT MIN(timestamp) FROM {TABLE}")[0]))
    periode_serie = "W" if (b["fin"] - debut_effectif).days <= 92 else "M"  # hebdomadaire sur 3 mois ou moins

    def serie(evts: pd.DataFrame, debut, fin) -> list[dict]:
        index = pd.period_range(debut, fin - timedelta(days=1), freq=periode_serie)
        cout = evts.groupby(evts["date"].dt.to_period(periode_serie))["cout"].sum().reindex(index, fill_value=0.0)
        return [{"debut": p.start_time.date().isoformat(), "cout": float(v)} for p, v in cout.items()]

    return {
        "periode": {"code": code, "libelle": b["libelle"], "libelle_precedente": prec["libelle"] if prec else None},
        "indicateurs": indicateurs(conn, b["debut"], b["fin"]),
        "indicateurs_precedents": indicateurs(conn, prec["debut"], prec["fin"]) if prec else None,
        "pannes_par_gamme": [
            {"gamme": g, "cause": c, "pannes": int(n)} for (g, c), n in ev.groupby(["gamme", "cause"]).size().items()
        ],
        "cout_serie": serie(ev, debut_effectif, b["fin"]),
        "cout_serie_precedente": serie(evenements(conn, prec["debut"], prec["fin"]), prec["debut"], prec["fin"]) if prec else [],
    }


def etat_parc(conn: duckdb.DuckDBPyConnection, seuil: float) -> list[dict]:
    """Dernier relevé de chaque machine : statut affiché, risque, criticité et zone d'action."""
    df = conn.execute(
        f"""SELECT t.id_machine, t.type_machine AS gamme, t.timestamp, t.etat,
                   p.probabilite_panne, p.niveau, p.cause_principale
            FROM {TABLE} t JOIN {TABLE_PREDICTIONS} p USING (id_machine, timestamp)
            QUALIFY ROW_NUMBER() OVER (PARTITION BY t.id_machine ORDER BY t.timestamp DESC) = 1
            ORDER BY t.id_machine"""
    ).fetchdf()
    machines = []
    for r in df.itertuples():
        crit = criticite(r.id_machine)
        st = statut(r.niveau, r.etat)
        machines.append(
            {
                "id_machine": r.id_machine,
                "numero": r.id_machine[-2:],
                "nom": nom_machine(r.id_machine),
                "gamme": r.gamme,
                "statut": st,
                "risque": float(r.probabilite_panne),
                "criticite": crit,
                "zone": zone(float(r.probabilite_panne), seuil, crit) if st != "en_panne" else None,
                "cause": r.cause_principale if isinstance(r.cause_principale, str) else None,
                "sante": sante(float(r.probabilite_panne)),
            }
        )
    return machines


@lru_cache(maxsize=1)
def limites_capteurs() -> dict[str, float]:
    """99e centile du parc sur la période d'entraînement : repère « au-dessus de la normale » de chaque capteur."""
    with connexion() as conn:
        colonnes = ", ".join(f"QUANTILE_CONT({c}, {0.01 if s == 'min' else 0.99})" for c, (_, _, s) in CAPTEURS_FICHE.items())
        valeurs = _ligne(conn, f"SELECT {colonnes} FROM {TABLE} WHERE timestamp < ?", [DATE_TEST])
    return {c: round(float(v), 1) for c, v in zip(CAPTEURS_FICHE, valeurs)}


def fiche_machine(conn: duckdb.DuckDBPyConnection, id_machine: str, seuil: float) -> dict | None:
    """Fiche de l'écran Machine (hors contributions SHAP, calculées par l'API qui détient le modèle)."""
    etat = next((m for m in etat_parc(conn, seuil) if m["id_machine"] == id_machine), None)
    if etat is None:
        return None
    fin = fin_donnees(conn)
    fin_excl = fin + PAS
    historique = evenements(conn, fin_excl - timedelta(days=730), fin_excl, id_machine)
    annee = indicateurs(conn, fin_excl - timedelta(days=365), fin_excl, id_machine)
    debut_30j = fin_excl - timedelta(days=30)
    agregats = ", ".join(f"{s.upper()}({c})" for c, (_, _, s) in CAPTEURS_FICHE.items())
    valeurs = _ligne(
        conn, f"SELECT {agregats}, SUM(panne) FROM {TABLE} WHERE id_machine = ? AND timestamp >= ?", [id_machine, debut_30j]
    )
    limites = limites_capteurs()
    capteurs = []
    for (colonne, (libelle, unite, sens)), valeur in zip(CAPTEURS_FICHE.items(), valeurs):
        hors_limite = valeur is not None and (valeur > limites[colonne] if sens == "max" else valeur < limites[colonne])
        capteurs.append(
            {
                "capteur": colonne,
                "libelle": libelle,
                "unite": unite,
                "valeur": valeur,
                "limite": limites[colonne],
                "hors_limite": hors_limite,
            }
        )
    return {
        **etat,
        "seuil": seuil,
        "historique": historique[["date", "cause", "duree_h", "cout"]].to_dict("records"),
        "douze_mois": annee,
        "capteurs_30j": capteurs,
        "pannes_30j": int(valeurs[-1] or 0),
    }


def serie_risque(conn: duckdb.DuckDBPyConnection, id_machine: str, jours: int) -> pd.DataFrame:
    """Risque au dernier relevé de chaque pas de 2 h (le dernier point est le risque actuel), délestages et pannes."""
    return conn.execute(
        f"""SELECT time_bucket(INTERVAL 2 HOUR, t.timestamp) AS timestamp,
                   arg_max(p.probabilite_panne, t.timestamp) AS risque,
                   MAX(t.indicateur_coupure) AS delestage,
                   MAX(t.panne) AS panne
            FROM {TABLE} t JOIN {TABLE_PREDICTIONS} p USING (id_machine, timestamp)
            WHERE t.id_machine = ?
              AND t.timestamp > (SELECT MAX(timestamp) FROM {TABLE} WHERE id_machine = ?) - ? * INTERVAL 1 DAY
            GROUP BY 1 ORDER BY 1""",
        [id_machine, id_machine, jours],
    ).fetchdf()


_MESURES = {
    "heures_depuis_changement_outil": ("Heures d'outil", "h"),
    "heures_depuis_maintenance": ("Depuis la dernière intervention", "h"),
    "vibration": ("Vibration", "mm/s"),
    "temperature_process": ("Temp. process", "°C"),
    "ecart_thermique": ("Écart thermique", "°C"),
    "couple": ("Couple", "Nm"),
    "concentration_poussiere": ("Poussière", "µg/m³"),
    "niveau_tension": ("Tension", "V"),
    "humidite": ("Humidité", "%"),
    "temperature_ambiante": ("Temp. atelier", "°C"),
    "vitesse_rotation": ("Vitesse", "tr/min"),
    "niveau_activite": ("Charge de production", ""),
    "precipitation_mm": ("Pluie", "mm"),
    "indicateur_coupure": ("Délestage", ""),
    "type_machine": ("Gamme", ""),
}
_SUFFIXES = {
    "_moyenne_24h": " moy. 24 h",
    "_max_24h": " max 24 h",
    "_ecart_type_24h": " écart-type 24 h",
    "_hausse_4h": " hausse 4 h",
}


def libelle_feature(feature: str) -> tuple[str, str]:
    """Libellé lisible et unité d'une feature du modèle (ex. vibration_max_24h -> « Vibration max 24 h », mm/s)."""
    for suffixe, texte in _SUFFIXES.items():
        if feature.endswith(suffixe):
            libelle, unite = _MESURES.get(feature.removesuffix(suffixe), (feature, ""))
            return libelle + texte, unite
    return _MESURES.get(feature, (feature, ""))
