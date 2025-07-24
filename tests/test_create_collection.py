from fastapi.testclient import TestClient
import uuid
import pytest
from server import app

client = TestClient(app)


@pytest.fixture
def collection_name():
    return f"test_{uuid.uuid4().hex}"


def test_create_collection(collection_name):
    # create random collection name
    # collection_name = f"test_{uuid.uuid4().hex}"
    response = client.put(
        f"/collections/{collection_name}",
        json={"embeddings_model": "text-embedding-3-small"}
    )

    # Assert that the response is successful (status code 200)
    assert response.status_code == 200
    assert "created" in response.json()["status"]

    response = client.delete(f"/collections/{collection_name}")


def test_create_existing_collection(collection_name):
    # create random collection name
    # collection_name = f"test_{uuid.uuid4().hex}"
    client.put(f"/collections/{collection_name}", json={"embeddings_model": "text-embedding-3-small"})

    # Try to create the same collection again
    response = client.put(
        f"/collections/{collection_name}",
        json={"embeddings_model": "text-embedding-3-small"}
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]
    response = client.delete(f"/collections/{collection_name}")
    assert response.status_code == 200
