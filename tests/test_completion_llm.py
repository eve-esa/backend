import pytest
from fastapi.testclient import TestClient
from server import app
import httpx
from src.config import Config

client = TestClient(app)
config = Config()


@pytest.mark.asyncio
async def test_completion_llm_real_api():
    request_data = {"query": "ESA is a space agency and"}

    async with httpx.AsyncClient() as async_client:
        response = await async_client.post(
            "http://127.0.0.1:8000/completion_llm",
            json=request_data,
            timeout=config.get_timeout(),
        )

    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}"

    json_response = response.json()

    assert "query" in json_response
    assert "response" in json_response
    assert json_response["query"] == request_data["query"], "Query mismatch"
    assert len(json_response["response"]) > 0, "Response should not be empty"
