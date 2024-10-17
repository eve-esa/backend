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
    response = client.post(
        "/create_collection", json={"collection_name": f"{collection_name}"}
    )

    # Assert that the response is successful (status code 200)
    assert response.status_code == 200
    assert response.json() == {
        "message": f"Collection '{collection_name}' created successfully"
    }

    response = client.delete(f"/delete_collection?collection_name={collection_name}")


def test_create_existing_collection(collection_name):
    # create random collection name
    # collection_name = f"test_{uuid.uuid4().hex}"
    client.post("/create_collection", json={"collection_name": f"{collection_name}"})

    # Try to create the same collection again
    response = client.post(
        "/create_collection", json={"collection_name": f"{collection_name}"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": f"Collection '{collection_name}' already exists"
    }
    response = client.delete(f"/delete_collection?collection_name={collection_name}")
    assert response.status_code == 200
