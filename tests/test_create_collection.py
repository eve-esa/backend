from fastapi.testclient import TestClient
import uuid
import pytest
from unittest.mock import patch, Mock
from server import app

client = TestClient(app)


@pytest.fixture
def collection_name():
    return f"test_{uuid.uuid4().hex}"


@patch("src.services.collection.VectorStoreManager")
def test_create_collection(mock_vector_store, collection_name):
    """Test collection creation with mocked vector store."""
    # Mock the vector store manager
    mock_manager = Mock()
    mock_manager.create_collection.return_value = True
    mock_manager.list_collections_names.return_value = []
    mock_vector_store.return_value = mock_manager

    response = client.put(
        f"/collections/{collection_name}",
        json={"embeddings_model": "text-embedding-3-small"},
    )

    # Assert that the response is successful (status code 200)
    assert response.status_code == 200
    assert "created" in response.json()["status"]

    # Clean up
    response = client.delete(f"/collections/{collection_name}")


@patch("src.services.collection.VectorStoreManager")
def test_create_existing_collection(mock_vector_store, collection_name):
    """Test creating an existing collection with mocked vector store."""
    # Mock the vector store manager
    mock_manager = Mock()
    mock_manager.create_collection.return_value = True
    # First call returns empty list, second call returns the collection name
    mock_manager.list_collections_names.side_effect = [[], [collection_name]]
    mock_vector_store.return_value = mock_manager

    # Create the collection first
    client.put(
        f"/collections/{collection_name}",
        json={"embeddings_model": "text-embedding-3-small"},
    )

    # Try to create the same collection again
    response = client.put(
        f"/collections/{collection_name}",
        json={"embeddings_model": "text-embedding-3-small"},
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]

    # Clean up
    response = client.delete(f"/collections/{collection_name}")
    assert response.status_code == 200
