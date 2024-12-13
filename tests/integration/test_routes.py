
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_predict_route():
    response = client.post(
        "/predict/",
        json={"features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]},
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_train_route():
    response = client.post("/train/")
    assert response.status_code == 200
    assert response.json()["message"] == "Model trained successfully!"
