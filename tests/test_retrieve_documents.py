import pytest
from fastapi.testclient import TestClient
from server import app
import os

client = TestClient(app)


@pytest.mark.asyncio
async def test_retrieve_documents_success():
    response = client.post(
        "/retrieve_documents",
        json={"query": "What is the european space agency?", "k": 3},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)  # Check if the response is a list
    assert len(response.json()) <= 3  # Check that the number of documents is at most k


# TODO: fix this test
# @pytest.mark.asyncio
# async def test_retrieve_documents_no_results():
#     response = client.post(
#         "/retrieve_documents",
#         json={
#             "query": "empty",
#             "k": 3,
#         },
#     )
#     assert response.status_code == 404
#     assert response.json()["detail"] == "No documents found."


@pytest.mark.asyncio
async def test_retrieve_documents_error():
    response = client.post(
        "/retrieve_documents",
        json={"query": "error_query", "k": 3, "score_threshold": 0.9},
    )
    assert response.status_code == 500
    assert "An error occurred" in response.json()["detail"]
