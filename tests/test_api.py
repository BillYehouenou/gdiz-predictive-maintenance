from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

payload_normal = {
    "temperature_ambiante": 26.0,
    "temperature_process": 40.0,
    "vitesse_rotation": 1200.0,
    "couple": 28.0,
    "usure_outil": 20.0,
    "niveau_activite": 0.5,
    "vibration": 1.2,
    "humidite": 70.0,
    "concentration_poussiere": 45.0,
    "niveau_tension": 225.0,
    "stabilite_tension": 82.0,
    "indicateur_pluie": 1,
    "indicateur_coupure": 0,
    "saison_benin": 1,
    "type_machine": "H",
    "encrassement": 35.0,
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
