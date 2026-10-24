import pandas as pd
import pytest

from src import interventions as itv


@pytest.fixture
def conn(tmp_path):
    chemin = tmp_path / "interventions.db"
    itv.initialiser(chemin)
    with itv.connexion(chemin) as c:
        yield c


def _id_a_traiter(conn) -> int:
    return int(itv.lister(conn, "a_traiter").iloc[0]["id"])


def test_historique_et_alertes_en_cours(conn):
    assert len(itv.lister(conn, "a_traiter")) > 0
    f = itv.fiabilite(conn)
    assert 0 < f["alertes_fondees"] < f["alertes_qualifiees"]


def test_circuit_planifier_puis_qualifier(conn):
    i = _id_a_traiter(conn)
    assert itv.planifier(conn, i, "TECH-01", "31/12 matin", "Entretien")
    assert next(t for t in itv.charge_techniciens(conn) if t["id"] == "TECH-01")["interventions_planifiees"] == 1
    assert itv.qualifier(conn, i, "corrige")
    ligne = conn.execute("SELECT statut, fondee, source FROM interventions WHERE id = ?", [i]).fetchone()
    assert ligne == ("qualifiee", True, "terrain")
    assert itv.rouvrir(conn, i)
    assert conn.execute("SELECT statut FROM interventions WHERE id = ?", [i]).fetchone()[0] == "planifiee"


def test_transitions_interdites(conn):
    i = _id_a_traiter(conn)
    assert not itv.qualifier(conn, i, "corrige")  # pas de constat sans intervention planifiée
    assert itv.qualifier(conn, i, "ecartee")  # mais une alerte peut être écartée
    with pytest.raises(ValueError):
        itv.planifier(conn, i, "TECH-99", "31/12", "Entretien")


def test_alertes_proches_fusionnees_et_fondees():
    ts = pd.date_range("2025-03-01", periods=400, freq="30min")
    niveau = ["normal"] * 400
    for k in [*range(10, 14), *range(30, 35), *range(200, 203)]:  # 2 alertes à 8 h d'écart, puis une isolée
        niveau[k] = "surveillance"
    panne = [0] * 400
    panne[60] = 1
    df = pd.DataFrame(
        {"id_machine": "M", "timestamp": ts, "niveau": niveau, "panne": panne, "mode_panne": ["TWF" if p else "" for p in panne]}
    )
    ep = itv.episodes_alerte(df)
    assert len(ep) == 2
    assert ep["fondee"].tolist() == [True, False]
