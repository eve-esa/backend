# tests/test_delete_collection.py

import pytest
from fastapi.testclient import TestClient
from server import app  # Adjust the import as needed for your FastAPI app

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_teardown():
    # Setup: Create a collection to test deletion
    response = client.post(
        "/create_collection", json={"collection_name": "test_collection"}
    )
    assert response.status_code == 200  # Ensure creation was successful
    yield  # This is where the testing happens

    # Teardown: Cleanup after tests
    response = client.delete("/delete_collection?collection_name=test_collection")
    if response.status_code == 200:
        assert response.json() == {
            "message": "Collection 'test_collection' deleted successfully."
        }
    else:
        # Optionally log or print if deletion fails
        print(f"Failed to delete collection: {response.json()}")


def test_delete_collection_success(setup_teardown):
    response = client.delete("/delete_collection?collection_name=test_collection")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Collection 'test_collection' deleted successfully."
    }


def test_delete_collection_not_found(setup_teardown):
    response = client.delete(
        "/delete_collection?collection_name=non_existing_collection"
    )
    assert response.status_code == 404
    assert response.json() == {
        "detail": "Collection 'non_existing_collection' does not exist."
    }
