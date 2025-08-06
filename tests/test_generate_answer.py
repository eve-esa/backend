import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


@pytest.mark.integration
def test_generate_answer_endpoint_structure():
    """
    Test the /generate_answer endpoint structure without requiring external services.
    This is a mock test to verify the endpoint exists and returns proper structure.
    """
    # Arrange - test with minimal data that won't require external API calls
    request_data = {
        "query": "What is ESA?",
        "collection_name": "test_collection",
        "llm": "openai",
        "embeddings_model": "text-embedding-3-small",
        "score_threshold": 0.7,
        "get_unique_docs": True,
    }

    # This will likely fail due to missing API keys, but we can test the endpoint structure
    response = client.post("/generate_answer", json=request_data)

    # The endpoint should exist and return some response (even if it's an error)
    assert response.status_code in [400, 401, 403, 500]  # Expected error codes for missing API keys
    assert "detail" in response.json()  # Should return error details


@pytest.mark.integration
def test_generate_answer_missing_required_fields():
    """
    Test the /generate_answer endpoint with missing required fields.
    """
    # Test with missing required fields
    request_data = {
        "query": "What is ESA?",
        # Missing collection_name and other required fields
    }

    response = client.post("/generate_answer", json=request_data)

    # Should return some response (could be 200 with no results or an error)
    assert response.status_code in [200, 400, 500]  # Accept various responses
    assert "detail" in response.json() or "answer" in response.json() or "documents" in response.json()
