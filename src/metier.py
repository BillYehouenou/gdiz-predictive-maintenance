"""
Le métier : chaîne de production, coûts et règles du superviseur (hypothèses de projet, lisibles et testées), puis les
calculs servis au tableau de bord depuis DuckDB : indicateurs par période, pannes, état du parc, fiche machine.
"""

from datetime import timedelta
from functools import lru_cache
from typing import Any, Literal, get_args

import duckdb
import pandas as pd

from src.config import CHEMIN_DB, PAS, TABLE, TABLE_PREDICTIONS
from src.donnees import DATE_TEST

# Chaîne de production : la criticité vient de la part de production perdue quand une machine s'arrête
CHAINE: list[dict[str, Any]] = [
    {"etape": "Préparation", "machine": "Cardeuse", "ids": [4, 11, 20, 37]},
    {"etape": "Filature", "machine": "Continu à filer", "ids": [1, 5, 7, 12, 16, 19, 22, 25, 28, 31, 34, 39, 43, 47]},
    {"etape": "Tissage", "machine": "Métier à tisser",
     "ids": [2, 3, 6, 9, 10, 13, 14, 17, 18, 21, 23, 24, 27, 29, 32, 35, 36, 38, 40, 42, 44, 45, 48, 49]},
    {"etape": "Finition", "machine": "Rameuse", "ids": [8, 15, 26, 30, 33, 41, 46, 50]},
]  # fmt: skip
_ETAPE = {n: etape for etape in CHAINE for n in etape["ids"]}
_PERTE_MAX = max(1 / len(etape["ids"]) for etape in CHAINE)

# Coûts (FCFA) : production perdue par heure d'arrêt, pièces par panne, tissu rebuté au crash
TAUX_TECHNICIEN, COUT_MATIERE = 3_250, 800  # par heure, par mètre
GAMMES: dict[str, dict[str, Any]] = {
    "L": {"capacite_m_h": 15, "marge_m": 1_200, "pieces": 90_000, "rebut_m": 5},
    "M": {"capacite_m_h": 20, "marge_m": 1_500, "pieces": 180_000, "rebut_m": 8},
    "H": {"capacite_m_h": 35, "marge_m": 2_000, "pieces": 500_000, "rebut_m": 15},
}
# Valeur du modèle : une panne annoncée devient un arrêt planifié court, sans rebut ; une alerte coûte une inspection
DUREE_INTERVENTION_PLANIFIEE_H, COUT_INSPECTION = 2.0, TAUX_TECHNICIEN  # une heure de technicien, machine en marche

CodePeriode = Literal["30j", "t4", "s2", "annee", "tout"]
PERIODES = get_args(CodePeriode)
CAPTEURS_FICHE = {  # colonne : libellé, unité, sens de l'alerte
    "vibration": ("Vibration max", "mm/s", "max"),
    "temperature_process": ("Temp. process max", "°C", "max"),
    "couple": ("Couple max", "Nm", "max"),
    "concentration_poussiere": ("Poussière max", "µg/m³", "max"),
    "niveau_tension": ("Tension min", "V", "min"),
}


def nom_machine(id_machine: str) -> str:
    return _ETAPE[int(id_machine[-2:])]["machine"]


def criticite(id_machine: str) -> float:
    """Part de la production perdue si la machine s'arrête (1 / machines de son étape), ramenée sur 10."""
    return round(1 / len(_ETAPE[int(id_machine[-2:])]["ids"]) / _PERTE_MAX * 10, 1)


def cout_heure_arret(gamme: str) -> float:
    return GAMMES[gamme]["capacite_m_h"] * GAMMES[gamme]["marge_m"] + TAUX_TECHNICIEN


def cout_panne(gamme: str, duree_h: float) -> float:
    """pièces + (production perdue + technicien) × durée réelle d'arrêt + tissu rebuté."""
    g = GAMMES[gamme]
    return g["pieces"] + cout_heure_arret(gamme) * duree_h + g["rebut_m"] * COUT_MATIERE


def gain_panne_evitee(gamme: str, duree_h: float) -> float:
    """Coût de la panne subie moins celui de l'intervention planifiée qui la remplace (pièces payées dans les deux cas)."""
    return cout_panne(gamme, duree_h) - (GAMMES[gamme]["pieces"] + cout_heure_arret(gamme) * DUREE_INTERVENTION_PLANIFIEE_H)


def sante(risque: float) -> int:
    return round(100 * (1 - risque) ** 4)


def statut(niveau: str, etat: str) -> str:
    """Statut affiché : une machine en réparation est en panne, quel que soit son risque."""
    return "en_panne" if etat == "panne" else {"eleve": "panne_predite", "surveillance": "surveillance"}.get(niveau, "normal")


def zone(risque: float, seuil: float, crit: float) -> str | None:
    """Au-dessus du seuil. Agir : risque ≥ 3 × seuil. Sinon planifier si criticité ≥ 4, surveiller en dessous."""
    if risque < seuil:
        return None
    return "agir" if risque >= 3 * seuil else "planifier" if crit >= 4 else "surveiller"


def connexion() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(CHEMIN_DB), read_only=True)


def _ligne(conn: duckdb.DuckDBPyConnection, sql: str, parametres: list | None = None) -> tuple:
    ligne = conn.execute(sql, parametres or []).fetchone()
    assert ligne is not None
    return ligne


def fin_donnees(conn: duckdb.DuckDBPyConnection) -> pd.Timestamp:
    return pd.Timestamp(_ligne(conn, f"SELECT MAX(timestamp) FROM {TABLE}")[0])


def libelles_periodes(fin: pd.Timestamp) -> dict[str, str]:
    """Libellé court de chaque période pour le sélecteur, daté par l'année des dernières données."""
    a = fin.year
    return {"30j": "30 j", "t4": f"T4 {a}", "s2": f"S2 {a}", "annee": str(a), "tout": f"{a - 1}-{a}"}


def bornes_periode(code: str, fin: pd.Timestamp, debut_donnees: pd.Timestamp | None = None) -> dict:
    """Début et fin (exclue) de la période, et de la période précédente de même nature ; « tout » part du premier relevé."""
    fin_excl, a = fin.normalize() + timedelta(days=1), fin.year
    if code == "tout":
        debut = debut_donnees if debut_donnees is not None else pd.Timestamp(a - 1, 1, 1)
        return {"debut": debut, "fin": fin_excl, "libelle": f"{a - 1}-{a}", "precedente": None}
    debut_30j = fin_excl - timedelta(days=30)
    periodes = {  # début, début de la précédente, libellés
        "30j": (debut_30j, debut_30j - timedelta(days=30), "30 derniers jours", "30 jours précédents"),
        "t4": (pd.Timestamp(a, 10, 1), pd.Timestamp(a, 7, 1), f"T4 {a}", f"T3 {a}"),
        "s2": (pd.Timestamp(a, 7, 1), pd.Timestamp(a, 1, 1), f"S2 {a}", f"S1 {a}"),
        "annee": (pd.Timestamp(a, 1, 1), pd.Timestamp(a - 1, 1, 1), str(a), str(a - 1)),
    }
    if code not in periodes:
        raise ValueError(f"Période inconnue : {code}")
    debut, debut_prec, libelle, libelle_prec = periodes[code]
    return {
        "debut": debut,
        "fin": fin_excl,
        "libelle": libelle,
        "precedente": {"debut": debut_prec, "fin": debut, "libelle": libelle_prec},
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


def indicateurs(
    conn: duckdb.DuckDBPyConnection, debut, fin, id_machine: str | None = None, ev: pd.DataFrame | None = None
) -> dict:
    """Disponibilité (hors arrêts planifiés et délestages), MTBF (heures de marche / pannes), coût, heures d'arrêt."""
    filtre = "AND id_machine = ?" if id_machine else ""
    marche, panne = _ligne(
        conn,
        f"""SELECT COUNT(*) FILTER (WHERE etat = 'marche'), COUNT(*) FILTER (WHERE etat = 'panne')
            FROM {TABLE} WHERE timestamp >= ? AND timestamp < ? {filtre}""",
        [debut, fin] + ([id_machine] if id_machine else []),
    )
    ev = evenements(conn, debut, fin, id_machine) if ev is None else ev
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
    debut_donnees, fin = (pd.Timestamp(v) for v in _ligne(conn, f"SELECT MIN(timestamp), MAX(timestamp) FROM {TABLE}"))
    b = bornes_periode(code, fin, debut_donnees)
    prec = b["precedente"]
    ev = evenements(conn, b["debut"], b["fin"])
    ev_prec = evenements(conn, prec["debut"], prec["fin"]) if prec else None
    periode_serie = "W" if (b["fin"] - max(b["debut"], debut_donnees)).days <= 92 else "M"  # hebdomadaire sur 3 mois ou moins

    def serie(evts: pd.DataFrame, debut, fin) -> list[dict]:
        index = pd.period_range(debut, fin - timedelta(days=1), freq=periode_serie)
        cout = evts.groupby(evts["date"].dt.to_period(periode_serie))["cout"].sum().reindex(index, fill_value=0.0)
        return [{"debut": p.start_time.date().isoformat(), "cout": float(v)} for p, v in cout.items()]

    return {
        "periode": {"code": code, "libelle": b["libelle"], "libelle_precedente": prec["libelle"] if prec else None},
        "indicateurs": indicateurs(conn, b["debut"], b["fin"], ev=ev),
        "indicateurs_precedents": indicateurs(conn, prec["debut"], prec["fin"], ev=ev_prec) if prec else None,
        "pannes_par_gamme": [
            {"gamme": g, "cause": c, "pannes": int(n)} for (g, c), n in ev.groupby(["gamme", "cause"]).size().items()
        ],
        "cout_serie": serie(ev, max(b["debut"], debut_donnees), b["fin"]),
        "cout_serie_precedente": serie(ev_prec, prec["debut"], prec["fin"]) if prec else [],
    }


def etat_parc(conn: duckdb.DuckDBPyConnection, seuil: float, id_machine: str | None = None) -> list[dict]:
    """
    Dernier relevé de chaque machine (ou d'une seule) : statut affiché, criticité et zone d'action. Le risque est celui du
    dernier relevé en marche, seul état noté par le modèle (une machine en panne garde le risque d'avant sa panne).
    """
    filtre = "WHERE id_machine = ?" if id_machine else ""
    df = conn.execute(
        f"""WITH derniers AS (SELECT id_machine, MAX(timestamp) AS timestamp FROM {TABLE} {filtre} GROUP BY 1)
            SELECT t.id_machine, t.type_machine AS gamme, t.timestamp, t.etat,
                   p.probabilite_panne, p.niveau, p.cause_principale
            FROM derniers JOIN {TABLE} t USING (id_machine, timestamp)
            ASOF LEFT JOIN {TABLE_PREDICTIONS} p ON p.id_machine = t.id_machine AND p.timestamp <= t.timestamp
            ORDER BY t.id_machine""",
        [id_machine] if id_machine else [],
    ).fetchdf()
    machines = []
    for r in df.itertuples():
        crit = criticite(r.id_machine)
        st = statut(r.niveau, r.etat)
        risque = float(r.probabilite_panne) if pd.notna(r.probabilite_panne) else 0.0
        machines.append(
            {
                "id_machine": r.id_machine,
                "numero": r.id_machine[-2:],
                "nom": nom_machine(r.id_machine),
                "gamme": r.gamme,
                "statut": st,
                "risque": risque,
                "criticite": crit,
                "zone": zone(risque, seuil, crit) if st != "en_panne" else None,
                "cause": r.cause_principale if isinstance(r.cause_principale, str) else None,
                "sante": sante(risque),
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
    etat = next(iter(etat_parc(conn, seuil, id_machine)), None)
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
                "sens": sens,
                "hors_limite": hors_limite,
            }
        )
    return {
        **etat,
        "seuil": seuil,
        "historique": historique[["date", "cause"]].to_dict("records"),
        "douze_mois": annee,
        "capteurs_30j": capteurs,
        "pannes_30j": int(valeurs[-1] or 0),
    }


def serie_risque(conn: duckdb.DuckDBPyConnection, id_machine: str, jours: int) -> pd.DataFrame:
    """
    Risque au dernier relevé en marche de chaque pas de 2 h (vide si la machine n'a pas tourné : arrêt, panne,
    délestage), délestages et pannes.
    """
    return conn.execute(
        f"""SELECT time_bucket(INTERVAL 2 HOUR, t.timestamp) AS timestamp,
                   arg_max(p.probabilite_panne, t.timestamp) FILTER (WHERE p.probabilite_panne IS NOT NULL) AS risque,
                   MAX(t.indicateur_coupure) AS delestage,
                   MAX(t.panne) AS panne
            FROM {TABLE} t LEFT JOIN {TABLE_PREDICTIONS} p USING (id_machine, timestamp)
            WHERE t.id_machine = ?
              AND t.timestamp > (SELECT MAX(timestamp) FROM {TABLE} WHERE id_machine = ?) - ? * INTERVAL 1 DAY
            GROUP BY 1 ORDER BY 1""",
        [id_machine, id_machine, jours],
    ).fetchdf()


# Libellés lisibles des features du modèle (ex. vibration_max_24h -> « Vibration max 24 h », mm/s)
_MESURES = {
    "heures_depuis_changement_outil": ("Heures d'outil", "h"),
    "heures_depuis_maintenance": ("Depuis la dernière intervention", "h"), "vibration": ("Vibration", "mm/s"),
    "temperature_process": ("Temp. process", "°C"), "ecart_thermique": ("Écart thermique", "°C"),
    "couple": ("Couple", "Nm"), "concentration_poussiere": ("Poussière", "µg/m³"), "niveau_tension": ("Tension", "V"),
    "humidite": ("Humidité", "%"), "temperature_ambiante": ("Temp. atelier", "°C"), "vitesse_rotation": ("Vitesse", "tr/min"),
    "niveau_activite": ("Charge de production", ""), "precipitation_mm": ("Pluie", "mm"), "indicateur_coupure": ("Délestage", ""),
    "type_machine": ("Gamme", ""),
}  # fmt: skip
_SUFFIXES = {
    "_moyenne_24h": " moy. 24 h",
    "_max_24h": " max 24 h",
    "_ecart_type_24h": " écart-type 24 h",
    "_hausse_4h": " hausse 4 h",
}


def libelle_feature(feature: str) -> tuple[str, str]:
    for suffixe, texte in _SUFFIXES.items():
        if feature.endswith(suffixe):
            libelle, unite = _MESURES.get(feature.removesuffix(suffixe), (feature, ""))
            return libelle + texte, unite
    return _MESURES.get(feature, (feature, ""))
