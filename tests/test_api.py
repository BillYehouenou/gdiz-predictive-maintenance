from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

payload_normal = {
    "ambient_temperature": 26.0,
    "process_temperature": 40.0,
    "rotational_speed": 1200.0,
    "torque": 28.0,
    "tool_wear": 20.0,
    "activity_level": 0.5,
    "vibration": 1.2,
    "humidity": 70.0,
    "dust_concentration": 45.0,
    "voltage_level": 225.0,
    "voltage_stability": 82.0,
    "rain_flag": 1,
    "power_loss_indicator": 0,
    "benin_season": 1,
    "machine_type": "H",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_input():
    r = client.post("/api/v1/predict", json=payload_normal)
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) >= {"prediction", "failure_probability", "status"}
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["status"] == "success"


def test_predict_missing_fields():
    r = client.post("/api/v1/predict", json={"ambient_temperature": 25.0})
    assert r.status_code == 422


def test_predict_invalid_machine_type_constraint():
    """activity_level hors [0,1] doit être rejeté par Pydantic."""
    bad_payload = {**payload_normal, "activity_level": 1.5}
    r = client.post("/api/v1/predict", json=bad_payload)
    assert r.status_code == 422
