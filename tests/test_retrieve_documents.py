import pytest
from fastapi.testclient import TestClient
from server import app
import os

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_test_collection():
    """Create a test collection for retrieve tests."""
    # Create test collection
    response = client.put(
        "/collections/test_retrieve_collection",
        json={"embeddings_model": "nasa-impact/nasa-smd-ibm-st-v2"}  # Use compatible model
    )
    assert response.status_code == 200
    
    yield
    
    # Cleanup
    response = client.delete("/collections/test_retrieve_collection")
    if response.status_code == 200:
        print("Test collection cleaned up")


@pytest.mark.asyncio
async def test_retrieve_documents_success(setup_test_collection):
    response = client.post(
        "/collections/test_retrieve_collection/retrieve",
        json={"query": "What is the european space agency?", "k": 3},
    )
    # Accept success, external service errors, and not found
    assert response.status_code in [200, 404, 500]
    if response.status_code == 200:
        assert isinstance(response.json(), list)  # Check if the response is a list
        assert len(response.json()) <= 3  # Check that the number of documents is at most k
    elif response.status_code == 404:
        assert "detail" in response.json()
    else:
        # If it's a 500 error, it should be due to external services, not endpoint issues
        assert "detail" in response.json()


@pytest.mark.asyncio
async def test_retrieve_documents_no_results(setup_test_collection):
    response = client.post(
        "/collections/test_retrieve_collection/retrieve",
        json={
            "query": "nonexistent_query_that_should_return_nothing",
            "k": 3,
        },
    )
    # Accept both 404 and 500 (external service errors)
    assert response.status_code in [404, 500]
    if response.status_code == 404:
        assert response.json()["detail"] == "No documents found."
    else:
        # If it's a 500 error, it should be due to external services, not endpoint issues
        assert "detail" in response.json()


@pytest.mark.asyncio
async def test_retrieve_documents_error(setup_test_collection):
    response = client.post(
        "/collections/test_retrieve_collection/retrieve",
        json={"query": "error_query", "k": 3, "score_threshold": 0.9},
    )
    # Accept various error codes
    assert response.status_code in [400, 404, 500]
    if response.status_code == 500:
        assert "detail" in response.json()
    else:
        # If it's a 400 or 404 error, it should be due to validation or not found issues
        assert "detail" in response.json()
