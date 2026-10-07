"""The classic message routes are deprecated in favour of the agentic ones.

``/messages`` and ``/stream_messages`` keep answering (the frontend still calls
them with FEATURE_AGENTIC_CHAT off), but the OpenAPI schema flags them so API
clients see the replacement before the routes go.
"""

import pytest

pytestmark = [pytest.mark.no_db, pytest.mark.asyncio]

_CLASSIC = (
    "/conversations/{conversation_id}/messages",
    "/conversations/{conversation_id}/stream_messages",
)
_AGENTIC = (
    "/conversations/{conversation_id}/generate-agentic",
    "/conversations/{conversation_id}/stream-generate-agentic",
)


async def test_openapi_marks_only_the_classic_routes_deprecated(async_client):
    response = await async_client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]

    for path in _CLASSIC:
        assert paths[path]["post"].get("deprecated") is True, path
    for path in _AGENTIC:
        assert not paths[path]["post"].get("deprecated", False), path


async def test_classic_docstrings_name_the_agentic_replacement(async_client):
    paths = (await async_client.get("/openapi.json")).json()["paths"]

    for classic, agentic in zip(_CLASSIC, _AGENTIC):
        assert agentic in paths[classic]["post"]["description"], classic
