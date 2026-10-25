import pandas as pd
import pytest

from src import metier
from src.metier import CHAINE, cout_heure_arret, cout_panne, criticite, nom_machine, sante, statut, zone


@pytest.fixture(scope="module")
def conn():
    with metier.connexion() as c:
        yield c


@pytest.mark.parametrize("code", metier.PERIODES)
def test_tableau_parc_pour_chaque_periode(conn, code):
    t = metier.tableau_parc(conn, code)
    ind = t["indicateurs"]
    assert 0.9 < ind["disponibilite"] <= 1
    assert ind["pannes"] == sum(p["pannes"] for p in t["pannes_par_gamme"])
    assert sum(p["cout"] for p in t["cout_serie"]) == pytest.approx(ind["cout"])
    assert (t["indicateurs_precedents"] is None) == (code == "tout")


def test_bornes_periode_precedente_de_meme_nature():
    b = metier.bornes_periode("t4", pd.Timestamp("2025-12-30 23:30"))
    assert (b["debut"], b["fin"]) == (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31"))
    assert b["precedente"]["libelle"] == "T3 2025"


def test_libelles_periodes_dates_par_les_donnees():
    libelles = metier.libelles_periodes(pd.Timestamp("2025-12-30 23:30"))
    assert list(libelles.values()) == ["30 j", "T4 2025", "S2 2025", "2025", "2024-2025"]
    assert list(libelles) == list(metier.PERIODES)


def test_evenements_duree_issue_de_l_etat(conn):
    ev = metier.evenements(conn, "2025-01-01", "2026-01-01", "GDIZ_TEX_M_20")
    assert len(ev) == 1
    assert ev.iloc[0]["cause"] == "TWF"
    assert ev.iloc[0]["cout"] == pytest.approx(cout_panne("M", ev.iloc[0]["duree_h"]))


def test_etat_parc(conn):
    machines = metier.etat_parc(conn, seuil=0.029)
    assert len(machines) == 50
    en_panne = [m for m in machines if m["statut"] == "en_panne"]
    assert all(m["zone"] is None for m in en_panne)


def test_fiche_machine(conn):
    fiche = metier.fiche_machine(conn, "GDIZ_TEX_M_20", seuil=0.029)
    assert fiche["nom"] == "Cardeuse"
    assert len(fiche["capteurs_30j"]) == 5
    assert metier.fiche_machine(conn, "INCONNUE", seuil=0.029) is None


def test_chaine_couvre_les_50_machines_une_seule_fois():
    numeros = [n for etape in CHAINE for n in etape["ids"]]
    assert sorted(numeros) == list(range(1, 51))


def test_criticite_inversement_proportionnelle_a_la_redondance():
    assert criticite("GDIZ_TEX_M_20") == 10.0  # 1 cardeuse sur 4
    assert criticite("GDIZ_TEX_M_30") == 5.0  # 1 rameuse sur 8
    assert criticite("GDIZ_TEX_M_23") == 1.7  # 1 métier à tisser sur 24
    assert nom_machine("GDIZ_TEX_M_20") == "Cardeuse"


def test_cout_panne_croit_avec_la_duree_et_la_gamme():
    assert cout_panne("M", 3) - cout_panne("M", 1) == pytest.approx(2 * cout_heure_arret("M"))
    assert cout_panne("H", 2) > cout_panne("M", 2) > cout_panne("L", 2)


def test_sante_et_statut():
    assert sante(0) == 100
    assert sante(0.298) == 24
    assert statut("eleve", "marche") == "panne_predite"
    assert statut("normal", "panne") == "en_panne"


@pytest.mark.parametrize(
    "risque,crit,attendu",
    [
        (0.01, 10, None),
        (0.03, 1.7, "surveiller"),
        (0.03, 5, "planifier"),
        (0.03, 10, "planifier"),
        (0.3, 1.7, "agir"),
        (0.3, 10, "agir"),
    ],
)
def test_zone(risque, crit, attendu):
    assert zone(risque, 0.029, crit) == attendu
