import pandas as pd
import pytest

from src import metier
from src.usine import cout_panne


@pytest.fixture(scope="module")
def conn():
    with metier.connexion() as c:
        yield c


@pytest.mark.parametrize("code", list(metier.PERIODES))
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
