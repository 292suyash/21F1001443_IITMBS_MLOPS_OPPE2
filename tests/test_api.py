from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_live_check():
    response = client.get("/live_check")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

def test_ready_check():
    response = client.get("/ready_check")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
