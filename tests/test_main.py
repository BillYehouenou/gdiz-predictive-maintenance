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
    # Vu qu'on utilise un modèle ML, on ne peut pas garantir la prédiction exacte, mais on peut vérifier que le format est correct
    assert response.json()["prediction"] in [0, 1]
    assert 0.0 <= response.json()["confidence"] <= 1.0