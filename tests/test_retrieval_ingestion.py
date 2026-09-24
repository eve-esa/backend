"""The model must not see chunk ids; the UI must keep them.

``RetrievalContextInterceptor`` sits on the MCP tool-call chain. Given the raw
``/retrieve`` response the retrieval tool sends back, it stashes the full
documents in the per-request retrieval context and rewrites the text block
the model reads to collection, text and descriptive metadata only. On
staging (2026-09-22) the model copied the Qdrant point ids from that block
into its answer, glued to the sentences.
"""

import json
import re
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from mcp.types import CallToolResult, ImageContent, TextContent

from src.services.agents.core.runner import _collect_retrieval_documents
from src.services.mcp.retrieval_context import (
    get_retrieval_context,
    reset_retrieval_context,
    retrieval_context,
    set_retrieval_context,
)
from src.services.mcp.retrieval_ingestion import (
    RetrievalContextInterceptor,
    slim_document_for_model,
)

pytestmark = pytest.mark.no_db

PRIVATE_ID = "f6378a16-e113-4bbc-8ba8-7fd23b4b7e65"
OWNER_ID = "6a7c730d403e5a0eac286f74"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _private_doc() -> dict:
    """Shape of an uploaded-document chunk, as staging returned it."""
    return {
        "id": PRIVATE_ID,
        "version": 142,
        "score": 0.6558165,
        "reranking_score": 0.9928785712542649,
        "collection_name": "RS",
        "payload": {
            "text": "Passive sensors gather radiation that is emitted or reflected.",
            "metadata": {
                "source": "remote-sensing.pdf",
                "source_name": "remote-sensing.pdf",
                "filename": "remote-sensing.pdf",
                "file_type": "application/pdf",
                "upload_time": "2026-09-22T12:58:00",
                "document_id": "6ab27b5c8bc8659543f544a2",
            },
            "user_id": OWNER_ID,
            "collection_id": "6ab27b3b8bc8659543f544a1",
        },
        "text": "Passive sensors gather radiation that is emitted or reflected.",
        "metadata": {
            "source": "remote-sensing.pdf",
            "source_name": "remote-sensing.pdf",
            "filename": "remote-sensing.pdf",
            "file_type": "application/pdf",
            "upload_time": "2026-09-22T12:58:00",
            "document_id": "6ab27b5c8bc8659543f544a2",
        },
    }


def _public_doc() -> dict:
    """Shape of a public knowledge base chunk (integer point id, rich payload)."""
    return {
        "id": 481257,
        "version": 7,
        "score": 0.71,
        "reranking_score": 0.88,
        "collection_name": "EVE open access",
        "payload": {
            "content": "SAR instruments emit microwave pulses and record the echo.",
            "title": "Active microwave remote sensing",
            "url": "https://doi.org/10.1000/example",
            "doi": "10.1000/example",
            "year": 2021,
            "journal": "Remote Sensing",
            "headers": ["2. Sensors"],
            "env": "prod",
            "file_path": "s3://bucket/papers/example.pdf",
            "pipeline_metadata": {"chunker": "v3"},
            "created_at": "2025-01-01",
            "last_update": "2025-02-01",
            "n_citations": 12,
        },
        "text": "SAR instruments emit microwave pulses and record the echo.",
        "metadata": {},
    }


def _response(*docs: dict) -> dict:
    return {
        "retrieved_docs": list(docs),
        "latencies": {"qdrant_retrieval": 0.2, "reranking": 0.4},
        "original_query": "active vs passive remote sensing",
        "requery": None,
    }


def _result(*blocks) -> CallToolResult:
    return CallToolResult(content=list(blocks))


def _text(payload) -> TextContent:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return TextContent(type="text", text=text)


def _request(name="retrieve", server_name="eve_retrieval"):
    return SimpleNamespace(name=name, server_name=server_name, args={})


def _handler(result):
    async def handler(request):
        return result

    return handler


@pytest.fixture
def with_context():
    ctx, token = set_retrieval_context()
    try:
        yield ctx
    finally:
        reset_retrieval_context(token)


async def _run(result, interceptor=None):
    interceptor = interceptor or RetrievalContextInterceptor()
    return await interceptor(_request(), _handler(result))


def _model_text(result: CallToolResult) -> str:
    assert len(result.content) == 1
    return result.content[0].text


class TestModelFacingText:
    @pytest.mark.asyncio
    async def test_ids_and_owner_fields_do_not_reach_the_model(self, with_context):
        out = await _run(_result(_text(_response(_private_doc(), _public_doc()))))

        text = _model_text(out)
        assert not UUID_RE.search(text)
        assert PRIVATE_ID not in text
        assert OWNER_ID not in text
        assert "481257" not in text
        for key in (
            "user_id",
            "collection_id",
            "document_id",
            "score",
            "reranking_score",
            "version",
            "env",
            "file_path",
            "pipeline_metadata",
            "upload_time",
            "latencies",
        ):
            assert f'"{key}"' not in text, key

    @pytest.mark.asyncio
    async def test_model_keeps_text_collection_and_descriptive_metadata(
        self, with_context
    ):
        out = await _run(_result(_text(_response(_private_doc(), _public_doc()))))

        slim = json.loads(_model_text(out))
        assert set(slim) == {"retrieved_docs", "original_query"}
        private, public = slim["retrieved_docs"]
        assert private == {
            "collection_name": "RS",
            "text": "Passive sensors gather radiation that is emitted or reflected.",
            "metadata": {
                "source": "remote-sensing.pdf",
                "source_name": "remote-sensing.pdf",
                "filename": "remote-sensing.pdf",
                "file_type": "application/pdf",
            },
        }
        assert public["collection_name"] == "EVE open access"
        assert public["text"].startswith("SAR instruments emit")
        assert public["metadata"] == {
            "title": "Active microwave remote sensing",
            "url": "https://doi.org/10.1000/example",
            "doi": "10.1000/example",
            "year": 2021,
            "journal": "Remote Sensing",
            "headers": ["2. Sensors"],
            "n_citations": 12,
        }

    @pytest.mark.asyncio
    async def test_text_is_sent_once(self, with_context):
        out = await _run(_result(_text(_response(_private_doc()))))

        assert _model_text(out).count("Passive sensors gather radiation") == 1

    def test_non_string_text_survives(self):
        doc = _public_doc()
        envelope = {"chunks": [{"passage": "Wiley passage"}]}
        doc["text"] = envelope
        doc["payload"]["content"] = envelope

        slim = slim_document_for_model(doc)

        assert slim["text"] == envelope

    @pytest.mark.asyncio
    async def test_structured_content_is_dropped(self, with_context):
        response = _response(_private_doc())
        result = CallToolResult(content=[_text(response)], structuredContent=response)

        out = await _run(result)

        assert out.structuredContent is None


class TestContext:
    @pytest.mark.asyncio
    async def test_full_documents_land_in_the_context(self, with_context):
        await _run(_result(_text(_response(_private_doc(), _public_doc()))))

        docs = with_context.documents
        assert [d["id"] for d in docs] == [PRIVATE_ID, "481257"]
        assert docs[0]["score"] == 0.6558165
        assert docs[0]["collection_name"] == "RS"
        assert docs[0]["payload"]["user_id"] == OWNER_ID
        assert docs[1]["payload"]["title"] == "Active microwave remote sensing"

    @pytest.mark.asyncio
    async def test_two_calls_accumulate(self, with_context):
        await _run(_result(_text(_response(_private_doc()))))
        await _run(_result(_text(_response(_public_doc()))))

        assert [d["id"] for d in with_context.documents] == [PRIVATE_ID, "481257"]

    @pytest.mark.asyncio
    async def test_no_context_means_passthrough(self):
        assert get_retrieval_context() is None
        result = _result(_text(_response(_private_doc())))

        out = await _run(result)

        assert out is result
        assert PRIVATE_ID in _model_text(out)


class TestLatencies:
    async def test_latencies_land_in_the_context_not_in_the_model_text(
        self, with_context
    ):
        payload = {
            "retrieved_docs": [],
            "latencies": {
                "query_embedding_latency": 0.12,
                "qdrant_retrieval_latency": 0.34,
            },
        }
        result = await _run(_result(_text(payload)))
        text = json.loads(_model_text(result))
        assert "latencies" not in text
        assert [c.latencies for c in with_context.calls] == [payload["latencies"]]
        assert with_context.calls[0].tool_name == "eve_retrieval_retrieve"

    def test_merge_sums_per_key_for_rag_tools_only(self):
        from src.services.agents.core.runner import _merge_retrieval_latencies
        from src.services.mcp.retrieval_context import RetrievalCall, retrieval_context

        with retrieval_context() as ctx:
            ctx.calls.append(
                RetrievalCall(
                    "eve_retrieval_retrieve",
                    latencies={
                        "query_embedding_latency": 0.1,
                        "qdrant_retrieval_latency": 0.2,
                        "reranking_latency": 0.05,
                    },
                )
            )
            ctx.calls.append(
                RetrievalCall(
                    "eve_retrieval_retrieve",
                    latencies={"qdrant_retrieval_latency": 0.3, "other": "x"},
                )
            )
            ctx.calls.append(
                RetrievalCall("other_search", latencies={"qdrant_retrieval_latency": 9})
            )
            merged = _merge_retrieval_latencies(
                {"total_latency": 1.0}, {"eve_retrieval_retrieve"}
            )
        assert merged["total_latency"] == 1.0
        assert merged["query_embedding_latency"] == 0.1
        assert merged["qdrant_retrieval_latency"] == 0.5
        assert merged["reranking_latency"] == 0.05

    def test_merge_without_context_is_a_no_op(self):
        from src.services.agents.core.runner import _merge_retrieval_latencies

        assert _merge_retrieval_latencies(
            {"total_latency": 1.0}, {"eve_retrieval_retrieve"}
        ) == {"total_latency": 1.0}


class TestPassthrough:
    @pytest.mark.asyncio
    async def test_error_payload_untouched(self, with_context):
        result = _result(_text({"error": "Qdrant unavailable", "retrieved_docs": None}))

        out = await _run(result)

        assert out is result
        assert with_context.documents == []

    @pytest.mark.asyncio
    async def test_other_tool_json_untouched(self, with_context):
        result = _result(_text({"lat": 41.8, "lon": 12.67, "id": "place-1"}))

        out = await _run(result)

        assert out is result

    @pytest.mark.asyncio
    async def test_plain_text_untouched(self, with_context):
        result = _result(_text("The capital of Italy is Rome."))

        out = await _run(result)

        assert out is result

    @pytest.mark.asyncio
    async def test_non_text_blocks_kept_in_place(self, with_context):
        image = ImageContent(type="image", data="aGk=", mimeType="image/png")
        result = _result(image, _text(_response(_private_doc())))

        out = await _run(result)

        assert out.content[0] is image
        assert PRIVATE_ID not in out.content[1].text

    @pytest.mark.asyncio
    async def test_non_call_tool_result_untouched(self, with_context):
        message = ToolMessage(content="raw", tool_call_id="c1")

        out = await _run(message)

        assert out is message

    @pytest.mark.asyncio
    async def test_reduction_error_fails_open(self, with_context, monkeypatch):
        def boom(_response):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "src.services.mcp.retrieval_ingestion.slim_retrieval_response_for_model",
            boom,
        )
        result = _result(_text(_response(_private_doc())))

        out = await _run(result)

        assert out is result
        assert PRIVATE_ID in _model_text(out)


class TestCollectRetrievalDocuments:
    """The runner reads Sources from the context, not from the reduced ToolMessage."""

    def _messages(self, content):
        return [
            AIMessage(
                content="",
                tool_calls=[{"name": "eve_retrieval_retrieve", "args": {}, "id": "c1"}],
            ),
            ToolMessage(content=content, tool_call_id="c1", name="eve_retrieval_retrieve"),
        ]

    @pytest.mark.asyncio
    async def test_documents_come_from_the_context_with_ids(self):
        with retrieval_context() as ctx:
            out = await _run(_result(_text(_response(_private_doc(), _public_doc()))))
            messages = self._messages(_model_text(out))

            documents, calls, errors = _collect_retrieval_documents(
                messages, {"eve_retrieval_retrieve"}
            )

        assert (calls, errors) == (1, 0)
        assert [d["id"] for d in documents] == [PRIVATE_ID, "481257"]
        assert documents[0]["payload"]["user_id"] == OWNER_ID
        assert len(ctx.documents) == 2

    @pytest.mark.asyncio
    async def test_context_documents_of_other_tools_are_ignored(self):
        with retrieval_context() as ctx:
            other = SimpleNamespace(name="search", server_name="other", args={})
            await RetrievalContextInterceptor()(
                other, _handler(_result(_text(_response(_public_doc()))))
            )
            assert ctx.calls[0].tool_name == "other_search"

            documents, calls, errors = _collect_retrieval_documents(
                [], {"eve_retrieval_retrieve"}
            )

        assert (documents, calls, errors) == ([], 0, 0)

    @pytest.mark.asyncio
    async def test_context_counts_the_call_when_the_tool_message_never_streamed(
        self,
    ):
        with retrieval_context():
            await _run(_result(_text(_response(_public_doc()))))

            documents, calls, errors = _collect_retrieval_documents(
                [], {"eve_retrieval_retrieve"}
            )

        assert (calls, errors) == (1, 0)
        assert [d["id"] for d in documents] == ["481257"]

    def test_without_context_the_tool_message_is_parsed(self):
        messages = self._messages(json.dumps(_response(_private_doc())))

        documents, calls, errors = _collect_retrieval_documents(
            messages, {"eve_retrieval_retrieve"}
        )

        assert (calls, errors) == (1, 0)
        assert [d["id"] for d in documents] == [PRIVATE_ID]

    def test_empty_context_falls_back_to_the_tool_message(self):
        messages = self._messages(json.dumps(_response(_public_doc())))

        with retrieval_context():
            documents, calls, _ = _collect_retrieval_documents(
                messages, {"eve_retrieval_retrieve"}
            )

        assert calls == 1
        assert [d["id"] for d in documents] == ["481257"]

    def test_error_payload_still_counts_as_error(self):
        messages = self._messages(json.dumps({"error": "boom"}))

        with retrieval_context():
            documents, calls, errors = _collect_retrieval_documents(
                messages, {"eve_retrieval_retrieve"}
            )

        assert (documents, calls, errors) == ([], 0, 1)
