from fastapi.testclient import TestClient
from server import app  # Replace with the actual import for your FastAPI app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
