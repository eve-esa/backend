import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


@pytest.mark.integration
def test_create_collection_real_service():
    """
    Test the /generate_answer endpoint with real Qdrant service.
    """
    # Arrange
    request_data = {
        "query": "What is ESA?",
        "collection_name": "test_llm4eo",
        "llm": "openai",
        "embeddings_model": "mistral-embed",
        "score_threshold": 0.7,
        "get_unique_docs": True,
    }

    response = client.post("/generate_answer", json=request_data)

    assert response.status_code == 200
    response_json = response.json()

    assert "answer" in response_json
    assert "documents" in response_json
    assert isinstance(response_json["documents"], list)

    assert len(response_json["documents"]) > 0
    assert isinstance(response_json["answer"], str)
    assert len(response_json["answer"]) > 0
