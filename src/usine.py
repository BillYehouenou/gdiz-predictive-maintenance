"""
L'usine vue par le superviseur : chaîne de production, coûts par gamme, techniciens, et les règles métier simples
(criticité, santé, coût d'une panne, zone d'action). Hypothèses de projet, réunies ici pour être lues et testées.
"""

from typing import Any

# Chaîne de production : la criticité vient de la part de production perdue quand une machine s'arrête
CHAINE: list[dict[str, Any]] = [
    {"etape": "Préparation", "machine": "Cardeuse", "ids": [4, 11, 20, 37]},
    {"etape": "Filature", "machine": "Continu à filer", "ids": [1, 5, 7, 12, 16, 19, 22, 25, 28, 31, 34, 39, 43, 47]},
    {
        "etape": "Tissage",
        "machine": "Métier à tisser",
        "ids": [2, 3, 6, 9, 10, 13, 14, 17, 18, 21, 23, 24, 27, 29, 32, 35, 36, 38, 40, 42, 44, 45, 48, 49],
    },
    {"etape": "Finition", "machine": "Rameuse", "ids": [8, 15, 26, 30, 33, 41, 46, 50]},
]
_ETAPE_PAR_NUMERO = {n: etape for etape in CHAINE for n in etape["ids"]}
_PERTE_MAX = max(1 / len(etape["ids"]) for etape in CHAINE)

# Coûts par gamme (FCFA) : production perdue par heure d'arrêt, pièces par panne, tissu rebuté au crash
TAUX_TECHNICIEN = 3_250  # par heure
COUT_MATIERE = 800  # par mètre
GAMMES: dict[str, dict[str, Any]] = {
    "L": {"libelle": "Gamme L", "capacite_m_h": 15, "marge_m": 1_200, "pieces": 90_000, "rebut_m": 5},
    "M": {"libelle": "Gamme M", "capacite_m_h": 20, "marge_m": 1_500, "pieces": 180_000, "rebut_m": 8},
    "H": {"libelle": "Gamme H", "capacite_m_h": 35, "marge_m": 2_000, "pieces": 500_000, "rebut_m": 15},
}
# Réglage des alertes : coût d'une fausse alerte et gain d'une panne annoncée
HEURES_FAUSSE_ALERTE = 1.0  # temps technicien perdu
HEURES_ARRET_EVITEES = 3.0  # arrêt évité quand la panne est annoncée (MTTR)

TECHNICIENS = [
    {"id": "TECH-01", "specialite": "Mécanique", "equipe": "matin"},
    {"id": "TECH-02", "specialite": "Mécanique", "equipe": "après-midi"},
    {"id": "TECH-03", "specialite": "Électrique", "equipe": "matin"},
    {"id": "TECH-04", "specialite": "Mécanique", "equipe": "nuit"},
    {"id": "TECH-05", "specialite": "Électrique", "equipe": "après-midi"},
    {"id": "TECH-06", "specialite": "Polyvalent", "equipe": "nuit"},
]
EQUIPES = {"matin": range(6, 14), "après-midi": range(14, 22), "nuit": list(range(22, 24)) + list(range(0, 6))}


def numero(id_machine: str) -> int:
    return int(id_machine.rsplit("_", 1)[-1])


def nom_machine(id_machine: str) -> str:
    return _ETAPE_PAR_NUMERO[numero(id_machine)]["machine"]


def criticite(id_machine: str) -> float:
    """Part de la production perdue si la machine s'arrête (1 / machines de son étape), ramenée sur 10."""
    perte = 1 / len(_ETAPE_PAR_NUMERO[numero(id_machine)]["ids"])
    return round(perte / _PERTE_MAX * 10, 1)


def cout_heure_arret(gamme: str) -> float:
    g = GAMMES[gamme]
    return g["capacite_m_h"] * g["marge_m"] + TAUX_TECHNICIEN


def cout_panne(gamme: str, duree_h: float) -> float:
    """pièces + (production perdue + technicien) × durée réelle d'arrêt + tissu rebuté."""
    g = GAMMES[gamme]
    return g["pieces"] + cout_heure_arret(gamme) * duree_h + g["rebut_m"] * COUT_MATIERE


def sante(risque: float) -> int:
    return round(100 * (1 - risque) ** 4)


def statut(niveau: str, etat: str) -> str:
    """Statut affiché : une machine en réparation est en panne, quel que soit son risque."""
    if etat == "panne":
        return "en_panne"
    return {"eleve": "panne_predite", "surveillance": "surveillance"}.get(niveau, "normal")


def zone(risque: float, seuil: float, crit: float) -> str | None:
    """Agir : risque ≥ 3 × seuil, ou criticité ≥ 7 au-dessus du seuil. Planifier : criticité ≥ 4. Sinon surveiller."""
    if risque < seuil:
        return None
    if risque >= 3 * seuil or crit >= 7:
        return "agir"
    return "planifier" if crit >= 4 else "surveiller"


def technicien_conseille(specialite: str, heure: int) -> str:
    """Premier technicien de la spécialité dans l'équipe en poste ; sinon le premier de la spécialité."""
    candidats = [t for t in TECHNICIENS if t["specialite"] in (specialite, "Polyvalent")]
    en_poste = [t for t in candidats if heure in EQUIPES[t["equipe"]]]
    return (en_poste or candidats)[0]["id"]
