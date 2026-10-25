import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

payload_normal = {
    "temperature_ambiante": 26.0,
    "temperature_process": 40.0,
    "vitesse_rotation": 1200.0,
    "couple": 28.0,
    "niveau_activite": 0.5,
    "vibration": 1.2,
    "humidite": 70.0,
    "concentration_poussiere": 45.0,
    "niveau_tension": 225.0,
    "precipitation_mm": 1.5,
    "indicateur_coupure": 0,
    "type_machine": "H",
    "heures_depuis_changement_outil": 800.0,
    "heures_depuis_maintenance": 200.0,
}


def test_sante():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_predire_entree_valide():
    r = client.post("/api/v1/predict", json=payload_normal)
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) >= {"prediction", "probabilite_panne", "status"}
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probabilite_panne"] <= 1.0
    assert data["status"] == "success"


def test_predire_champs_manquants():
    r = client.post("/api/v1/predict", json={"temperature_ambiante": 25.0})
    assert r.status_code == 422


def test_predire_contrainte_invalide():
    """niveau_activite hors [0,1] doit être rejeté par Pydantic."""
    payload_invalide = {**payload_normal, "niveau_activite": 1.5}
    r = client.post("/api/v1/predict", json=payload_invalide)
    assert r.status_code == 422


def test_predire_lot():
    r = client.post("/api/v1/predict/batch", json=[payload_normal, payload_normal])
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    for item in data:
        assert item["prediction"] in [0, 1]
        assert 0.0 <= item["probabilite_panne"] <= 1.0


def test_predire_lot_identique_aux_predictions_unitaires():
    """Le lot (une seule prédiction) donne exactement les résultats des appels un par un, machines connues comprises."""
    lot = [
        {**payload_normal, "id_machine": "GDIZ_TEX_M_20"},
        {**payload_normal, "id_machine": "GDIZ_TEX_M_20", "vibration": 4.0},
        {**payload_normal, "id_machine": "GDIZ_TEX_M_03"},
        payload_normal,
    ]
    unitaires = [client.post("/api/v1/predict", json=r).json() for r in lot]
    assert client.post("/api/v1/predict/batch", json=lot).json() == unitaires


def test_predire_lot_vide():
    r = client.post("/api/v1/predict/batch", json=[])
    assert r.status_code == 200
    assert r.json() == []


def test_info_modele():
    r = client.get("/api/v1/info")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["model_version"], str)
    assert 0.0 <= data["threshold"] <= 1.0


def test_recharger_modele():
    r = client.post("/api/v1/reload")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert 0.0 <= data["threshold"] <= 1.0


@pytest.mark.parametrize("periode", ["30j", "t4", "s2", "annee", "tout"])
def test_parc(periode):
    r = client.get("/api/v1/parc", params={"periode": periode})
    assert r.status_code == 200
    assert r.json()["periode"]["code"] == periode


def test_parc_periode_inconnue():
    assert client.get("/api/v1/parc", params={"periode": "hier"}).status_code == 422


def test_etat_du_parc():
    etat = client.get("/api/v1/etat").json()
    assert len(etat["machines"]) == 50
    assert sum(etat["comptes"].values()) == 50


def test_fiche_machine():
    fiche = client.get("/api/v1/machines/GDIZ_TEX_M_20").json()
    assert fiche["nom"] == "Cardeuse"
    assert fiche["contributions"] and all(c["contribution"] > 0 for c in fiche["contributions"])
    assert client.get("/api/v1/machines/INCONNUE").status_code == 404


def test_risque_machine():
    r = client.get("/api/v1/machines/GDIZ_TEX_M_20/risque", params={"jours": 7})
    assert r.status_code == 200
    assert len(r.json()) == 7 * 12  # pas de 2 h
    assert client.get("/api/v1/machines/INCONNUE/risque").status_code == 404


def test_parc_refuse_une_periode_inconnue():
    assert client.get("/api/v1/parc?periode=hier").status_code == 422


def test_capteurs_donnent_leur_sens():
    capteurs = client.get("/api/v1/machines/GDIZ_TEX_M_20").json()["capteurs_30j"]
    assert {c["capteur"]: c["sens"] for c in capteurs}["niveau_tension"] == "min"
