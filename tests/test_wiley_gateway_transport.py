"""Wiley failures through the real Streamable HTTP transport.

A local HTTP server stands in for the Scholar Gateway (no internet): it answers
every MCP request with a fixed error status, or never answers. These tests prove
that a gateway failure costs the turn its Wiley documents only: the Qdrant
documents stay, nothing raises, and the error row names the HTTP status.
"""

import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas.generation_request import GenerationRequest
from src.services import generate_answer
from src.services import mcp_client_service as svc
from src.services.mcp_client_service import MultiServerMCPClientService

pytestmark = pytest.mark.no_db


class _GatewayStub:
    """Answers every request with ``status``; ``status=None`` never answers."""

    def __init__(self, status):
        self.status = status
        self.release = threading.Event()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                if stub.status is None:
                    stub.release.wait(10)
                    return
                body = b'{"detail": "stub"}'
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_POST = _answer
            do_GET = _answer
            do_DELETE = _answer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def error_rows():
    error_logger = MagicMock()
    error_logger.log_error = AsyncMock()
    with patch.object(generate_answer, "get_error_logger", return_value=error_logger):
        yield error_logger


@pytest.fixture
def gateway_at(monkeypatch):
    """Point the config.yaml Wiley server at a local URL."""

    def _point(url):
        MultiServerMCPClientService._shared_instance = None
        monkeypatch.setattr(
            svc.Config,
            "get_mcp_servers",
            lambda self: {
                generate_answer.WILEY_MCP_SERVER_NAME: {
                    "transport": "streamable_http",
                    "url": url,
                    "enabled": True,
                }
            },
        )

    yield _point
    MultiServerMCPClientService._shared_instance = None


QDRANT_DOC = {
    "id": "qdrant-1",
    "score": 0.8,
    "text": "Sentinel-2 carries the MultiSpectral Instrument.",
    "metadata": {"title": "Sentinel-2"},
    "collection_name": "qwen-512-filtered",
}


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_gateway_http_error_returns_no_documents_and_an_error_row(
    status, gateway_at, error_rows
):
    with _GatewayStub(status) as stub:
        gateway_at(stub.url)
        documents, _ = await generate_answer.get_mcp_context(
            GenerationRequest(query="ocean heat content")
        )

    assert documents == []
    error_rows.log_error.assert_awaited_once()
    assert (
        error_rows.log_error.await_args.kwargs["description"]
        == f"MCP retrieval failed: HTTP {status}"
    )


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_gateway_http_error_keeps_the_qdrant_documents(
    status, gateway_at, error_rows
):
    request = GenerationRequest(
        query="ocean heat content",
        k=5,
        public_collections=["qwen-512-filtered", "Wiley AI Gateway"],
    )
    with _GatewayStub(status) as stub, patch.object(
        generate_answer, "VectorStoreManager", MagicMock()
    ), patch.object(
        generate_answer,
        "get_rag_context",
        AsyncMock(return_value=([dict(QDRANT_DOC)], {})),
    ), patch.object(
        generate_answer,
        "rerank_candidates",
        AsyncMock(return_value=[{"index": 0, "reranking_score": 0.9}]),
    ):
        gateway_at(stub.url)
        _context, results, _lat, formated = await generate_answer.setup_rag_and_context(
            request
        )

    assert [r["id"] for r in results] == ["qdrant-1"]
    assert [r["id"] for r in formated] == ["qdrant-1"]


async def test_gateway_that_never_answers_hits_the_total_deadline(
    gateway_at, error_rows, monkeypatch
):
    monkeypatch.setattr(generate_answer, "WILEY_MCP_TIMEOUT_S", 0.5)
    with _GatewayStub(None) as stub:
        gateway_at(stub.url)
        started = time.perf_counter()
        documents, _ = await generate_answer.get_mcp_context(
            GenerationRequest(query="ocean heat content")
        )
        elapsed = time.perf_counter() - started

    assert documents == []
    assert elapsed < 5
    error_rows.log_error.assert_awaited_once()
    assert "timed out" in error_rows.log_error.await_args.kwargs["description"]


async def test_a_real_cancellation_still_propagates(gateway_at, error_rows):
    with _GatewayStub(None) as stub:
        gateway_at(stub.url)
        task = asyncio.create_task(
            generate_answer.get_mcp_context(GenerationRequest(query="q"))
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    error_rows.log_error.assert_not_awaited()
