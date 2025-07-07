# tests/test_delete_collection.py

import pytest
from fastapi.testclient import TestClient
from server import app  # Adjust the import as needed for your FastAPI app

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_teardown():
    # Setup: Create a collection to test deletion
    response = client.put(
        "/collections/test_collection",
        json={"embeddings_model": "text-embedding-3-small"}
    )
    assert response.status_code == 200  # Ensure creation was successful
    yield  # This is where the testing happens

    # Teardown: Cleanup after tests
    response = client.delete("/collections/test_collection")
    if response.status_code == 200:
        assert "deleted successfully" in response.json()["message"]
    else:
        # Optionally log or print if deletion fails
        print(f"Failed to delete collection: {response.json()}")


def test_delete_collection_success(setup_teardown):
    response = client.delete("/collections/test_collection")
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]


def test_delete_collection_not_found(setup_teardown):
    response = client.delete(
        "/collections/non_existing_collection"
    )
    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]
