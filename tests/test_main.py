from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


class FakeModel:
    def predict(self, data):
        return [0]

    def predict_proba(self, data):
        return [[0.999, 0.001]]


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_successful_prediction(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())

    features = {feature: 0.0 for feature in main.FEATURES}

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 0
    assert response.json()["label"] == "Legitimate"
    assert response.json()["fraud_probability"] == 0.001


def test_missing_feature_returns_400(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())

    features = {feature: 0.0 for feature in main.FEATURES}
    features.pop("Amount")

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 400
    assert "Amount" in response.json()["detail"]["missing_features"]
