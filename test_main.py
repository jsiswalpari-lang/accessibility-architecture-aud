from fastapi.testclient import TestClient
from unittest.mock import patch
import importlib

class DummyModel:
    def predict_proba(self, data):
        return [[0.2, 0.3, 0.5]]

with patch("joblib.load", return_value=DummyModel()):
    import main

client = TestClient(main.app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_predict():
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    assert "probabilities" in response.json()

def test_invalid_input():
    response = client.post(
        "/predict",
        json={"features": ["invalid"]}
    )
    assert response.status_code == 422
