"""Wiley retrieval through the Scholar Gateway v2 connector.

The network is mocked at the MCP transport: these tests prove what the backend
sends (URL, X-API-Key header, tool name and arguments) and how it reads the v2
response envelope, not that the live gateway answers.
"""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, TextContent

from src.constants import WILEY_PUBLIC_COLLECTIONS
from src.schemas.generation_request import GenerationRequest
from src.services import generate_answer
from src.services import mcp_client_service as svc
from src.services.mcp_client_service import MultiServerMCPClientService

pytestmark = pytest.mark.no_db

V2_URL = "https://custom-connector-v2.scholargateway.ai/mcp"
API_KEY = "test-v2-key"


def _v2_envelope(n: int = 3) -> dict:
    return {
        "success": True,
        "query": "ocean heat content",
        "count": n,
        "totalResults": 120,
        "queryId": "q-1",
        "executionTimeMs": 412,
        "allowance": {"remaining": 99},
        "results": [
            {
                "doi": f"10.1029/2020GL{i:06d}",
                "id": f"10.1029/2020GL{i:06d}#{i}",
                "text": f"chunk {i} about ocean heat content",
                "chunk_index": i,
                "rrf_score": 0.9 - i / 10,
                "metadata": {
                    "additionalMetadata": {
                        "title": f"Paper {i}",
                        "journalTitle": "Geophysical Research Letters",
                        "citationLine": f"GRL {i}",
                        "link": f"https://doi.org/10.1029/2020GL{i:06d}",
                        "publicationDate": "2021-03-01",
                        "authors": ["A. Author"],
                        "abstract": "abstract",
                        "isRetracted": False,
                    }
                },
            }
            for i in range(n)
        ],
    }


class _Recorder:
    def __init__(self, envelope: dict | None = None, text: str | None = None):
        self.transport_calls: list[dict] = []
        self.tool_calls: list[tuple[str, dict]] = []
        self.session_kwargs: list[dict] = []
        body = text if text is not None else json.dumps(envelope or _v2_envelope())
        self.result = CallToolResult(content=[TextContent(type="text", text=body)])

    def transport(self):
        recorder = self

        @asynccontextmanager
        async def fake_streamablehttp_client(url, headers=None, **kwargs):
            recorder.transport_calls.append(
                {"url": url, "headers": dict(headers or {}), **kwargs}
            )
            yield (MagicMock(), MagicMock(), lambda: None)

        return fake_streamablehttp_client

    def session(self):
        recorder = self

        class FakeSession:
            def __init__(self, read, write, **kwargs):
                recorder.session_kwargs.append(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def initialize(self):
                return None

            async def call_tool(self, name, arguments=None):
                recorder.tool_calls.append((name, dict(arguments or {})))
                return recorder.result

        return FakeSession


@pytest.fixture
def fresh_shared():
    MultiServerMCPClientService._shared_instance = None
    yield
    MultiServerMCPClientService._shared_instance = None


@pytest.fixture
def no_error_rows():
    error_logger = MagicMock()
    error_logger.log_error = AsyncMock()
    with patch.object(generate_answer, "get_error_logger", return_value=error_logger):
        yield error_logger


def _patched(recorder: _Recorder, key: str = API_KEY):
    return (
        patch.object(svc, "WILEY_AUTH_TOKEN", key),
        patch.object(svc, "streamablehttp_client", recorder.transport()),
        patch.object(svc, "ClientSession", recorder.session()),
    )


@pytest.mark.parametrize(
    "envelope",
    [
        {"success": False, "error": "allowance exhausted", "allowance": {"remaining": 0}},
        {"success": False},
    ],
)
async def test_success_false_is_a_failure_not_a_document(
    envelope, fresh_shared, no_error_rows
):
    recorder = _Recorder(envelope=envelope)
    key_patch, transport_patch, session_patch = _patched(recorder)
    with key_patch, transport_patch, session_patch:
        documents, _ = await generate_answer.get_mcp_context(
            GenerationRequest(query="q")
        )

    assert documents == []
    no_error_rows.log_error.assert_awaited_once()
    kwargs = no_error_rows.log_error.await_args.kwargs
    assert kwargs["description"] == "MCP retrieval failed: success false"
    if "error" in envelope:
        assert str(kwargs["error"]) == "allowance exhausted"


def test_config_points_the_wiley_server_at_v2():
    servers = svc.Config().get_mcp_servers()
    assert servers["eve-mcp-demo"]["url"] == V2_URL
    assert servers["eve-mcp-demo"]["transport"] == "streamable_http"


async def test_call_sends_x_api_key_and_the_v2_tool_and_args(
    fresh_shared, no_error_rows
):
    recorder = _Recorder()
    key_patch, transport_patch, session_patch = _patched(recorder)
    request = GenerationRequest(
        query="ocean heat content",
        k=10,
        score_threshold=0.7,
        year=[2015, 2024],
        public_collections=["Wiley AI Gateway"],
    )
    with key_patch, transport_patch, session_patch, patch(
        "httpx.AsyncClient", side_effect=AssertionError("no token exchange")
    ):
        await generate_answer.get_mcp_context(request)

    assert len(recorder.transport_calls) == 1
    call = recorder.transport_calls[0]
    assert call["url"] == V2_URL
    assert call["headers"] == {"X-API-Key": API_KEY}
    assert "Authorization" not in call["headers"]
    assert recorder.tool_calls == [
        (
            "search_wiley_fulltext",
            {
                "query": "ocean heat content",
                "topN": 20,
                "includeRetractedContent": False,
                "start_year": 2015,
                "end_year": 2024,
            },
        )
    ]
    # The deadline is one total budget in get_mcp_context, not per request.
    assert recorder.session_kwargs == [{}]


async def test_v2_envelope_becomes_wiley_documents(fresh_shared, no_error_rows):
    recorder = _Recorder(envelope=_v2_envelope(3))
    key_patch, transport_patch, session_patch = _patched(recorder)
    request = GenerationRequest(query="ocean heat content", k=2)
    with key_patch, transport_patch, session_patch:
        documents, latencies = await generate_answer.get_mcp_context(request)

    assert recorder.tool_calls[0][1]["topN"] == 4
    assert len(documents) == 3
    assert [d["doi"] for d in documents] == [
        "10.1029/2020GL000000",
        "10.1029/2020GL000001",
        "10.1029/2020GL000002",
    ]
    assert all(d["collection_name"] == "Wiley AI Gateway" for d in documents)
    assert all(d["text"].startswith("chunk ") for d in documents)
    assert latencies["mcp_retrieval_latency"] is not None
    no_error_rows.log_error.assert_not_awaited()


async def test_auth_error_is_logged_once_and_not_retried(fresh_shared, no_error_rows):
    recorder = _Recorder(text="401 Unauthorized: invalid API key")
    key_patch, transport_patch, session_patch = _patched(recorder)
    with key_patch, transport_patch, session_patch:
        documents, _ = await generate_answer.get_mcp_context(
            GenerationRequest(query="q")
        )

    assert documents == []
    assert len(recorder.tool_calls) == 1
    no_error_rows.log_error.assert_awaited_once()
    assert no_error_rows.log_error.await_args.kwargs["description"] == "MCP auth error"


async def test_missing_key_never_reaches_the_network(fresh_shared):
    recorder = _Recorder()
    key_patch, transport_patch, session_patch = _patched(recorder, key="")
    with key_patch, transport_patch, session_patch:
        client = MultiServerMCPClientService.get_shared()
        result = await client.call_tool_on_server(
            "eve-mcp-demo", "search_wiley_fulltext", {"query": "q"}
        )

    assert result["is_error"] is True
    assert "WILEY_AUTH_TOKEN" in result["error"]
    assert recorder.transport_calls == []


def test_tool_listing_client_carries_the_key_only_for_wiley():
    with patch.object(svc, "WILEY_AUTH_TOKEN", API_KEY), patch.object(
        svc, "MultiServerMCPClient"
    ) as client_cls:
        MultiServerMCPClientService(
            server_configs={
                "eve-mcp-demo": {"transport": "streamable_http", "url": V2_URL},
                "other": {
                    "transport": "streamable_http",
                    "url": "https://mcp.example/mcp",
                },
            }
        )

    connections = client_cls.call_args.args[0]
    assert connections["eve-mcp-demo"]["headers"] == {"X-API-Key": API_KEY}
    assert "headers" not in connections["other"]


def test_configured_api_key_header_wins():
    with patch.object(svc, "WILEY_AUTH_TOKEN", API_KEY):
        headers = svc.with_wiley_api_key(V2_URL, {"x-api-key": "from-config"})
    assert headers == {"x-api-key": "from-config"}


def test_lookalike_host_gets_no_key():
    with patch.object(svc, "WILEY_AUTH_TOKEN", API_KEY):
        assert svc.with_wiley_api_key("https://scholargateway.ai.evil.example/mcp", {}) == {}
        assert svc.with_wiley_api_key("https://notscholargateway.ai/mcp", {}) == {}


def test_wiley_catalog_entry_describes_the_v2_portfolio():
    (entry,) = WILEY_PUBLIC_COLLECTIONS
    assert entry["name"] == "Wiley AI Gateway"
    assert entry["description"] == (
        "Peer-reviewed full text from Wiley's portfolio of more than 2,000 journals, "
        "across the sciences, medicine, engineering, business, law and the humanities. "
        "Coverage follows Wiley's publishing portfolio, so disciplines are not equally "
        "represented. Search covers article text only, not figures, images or "
        "supplementary data. Retracted articles are excluded."
    )
    assert "num_documents" not in entry
    assert "67" not in entry["description"]
