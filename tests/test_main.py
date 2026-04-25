from fastapi.testclient import TestClient
from app.main import app

# On crée un client qui simule des requêtes HTTP sans lancer le serveur
client = TestClient(app)

def test_read_root():
    """Vérifie que l'accueil de l'API fonctionne"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de Maintenance Prédictive opérationnelle"}

def test_prediction_logic():
    """Vérifie qu'une température élevée déclenche bien une alerte"""
    payload = {
        "air_temperature": 300.0,
        "process_temperature": 320.0, # +20 degrés de diff
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 200 # Usure élevée
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    # Selon notre logique dans utils.py, ça doit être une panne (1)
    assert response.json()["prediction"] == 1
    assert "Danger" in response.json()["status"]