import pytest
from fastapi.testclient import TestClient

from app.main import EXAMPLE_PIXELS, app
from app.train import MODEL_PATH, train_model


@pytest.fixture(scope="module")
def client():
    if not MODEL_PATH.exists():
        train_model()
    return TestClient(app)


def test_training_accuracy():
    metrics = train_model()
    assert metrics["accuracy"] >= 0.9


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info(client):
    data = client.get("/model-info").json()
    assert data["accuracy"] >= 0.9
    assert data["test_size"] > 0


def test_sample(client):
    data = client.get("/sample").json()
    assert len(data["pixels"]) == 64
    assert 0 <= data["true_digit"] <= 9


def test_predict_example(client):
    response = client.post("/predict", json={"pixels": EXAMPLE_PIXELS})
    assert response.status_code == 200
    data = response.json()
    assert data["digit"] == 0
    assert 0 <= data["confidence"] <= 1


def test_predict_wrong_length(client):
    assert client.post("/predict", json={"pixels": [1.0] * 10}).status_code == 422


def test_predict_out_of_range(client):
    assert client.post("/predict", json={"pixels": [20.0] * 64}).status_code == 422


def test_predict_batch(client):
    sample = client.get("/sample").json()
    response = client.post("/predict-batch", json=[{"pixels": EXAMPLE_PIXELS}, {"pixels": sample["pixels"]}])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_predict_batch_empty(client):
    assert client.post("/predict-batch", json=[]).status_code == 400


def test_metrics_exposed(client):
    client.post("/predict", json={"pixels": EXAMPLE_PIXELS})
    text = client.get("/metrics").text
    assert "model_predictions_total" in text
    assert "app_requests_total" in text
    assert "model_prediction_confidence_bucket" in text
    assert "model_test_accuracy" in text
