import pytest
from fastapi.testclient import TestClient
from api.main import app, SessionLocal, Dataset

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == ["OK", 200]

def test_generate_and_db():
    # Génère un dataset
    response = client.post("/generate", json={"n_samples": 10})
    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert data["n_samples"] == 10

    # Vérifie que le dataset est bien en base
    db = SessionLocal()
    dataset = db.query(Dataset).filter(Dataset.id == data["dataset_id"]).first()
    db.close()
    assert dataset is not None
    assert len(dataset.data) > 0

def test_retrain():
    response = client.post("/retrain")
    assert response.status_code == 200

def test_predict_value():
    # S'assure qu'il y a au moins un dataset
    client.post("/generate", json={"n_samples": 10})
    response = client.get("/predict")
    assert response.status_code == 200
    data = response.json()
    for i in data["prediction"]:
        assert i == 1 or i == 0


def test_generate_hour_sign():
    response = client.post("/generate", json={"n_samples": 5})
    data = response.json()
    assert "hour" in data
    assert "columns_inverted" in data
    assert isinstance(data["hour"], int)
    assert isinstance(data["columns_inverted"], bool)

