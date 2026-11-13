"""Inline citations behind FEATURE_INLINE_CITATIONS (sources standard, A7).

Covers the three pieces and their seams: Source descriptors from retrieval
records and STAC items, the handles the retrieval interceptor hands the model,
and the runner rewriting cited handles into ``[n]`` markers, streaming,
persisting and adapting them. With the flag off every frame and the stored
message stay as they were.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent

from src import config
from src.services.agui.adapter import AguiAdapter, RunContext, encode_sse
from src.services.mcp.retrieval_context import (
    get_retrieval_context,
    reset_retrieval_context,
    set_retrieval_context,
)
from src.services.mcp.retrieval_ingestion import (
    RetrievalContextInterceptor,
    slim_retrieval_response_for_model,
)
from src.services.sources.citations import (
    CITATION_INSTRUCTION,
    CitationRewriter,
    SourceRegistry,
    marker_spans,
)
from src.services.sources.descriptors import (
    LICENSED_PASSAGE_MAX_CHARS,
    PASSAGE_MAX_CHARS,
    cut_passage,
    public_source,
    source_from_document,
    source_from_stac_item,
)
from src.utils.helpers import extract_document_data
from tests.agui.validator import validate_transcript
from tests.test_agent_step_frames import RETRIEVE, _answer, _call, _Graph, _of, _result, _tool
from tests.test_agent_step_frames import _run as run_runner

pytestmark = pytest.mark.no_db

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
OWNER_ID = "6a7c730d403e5a0eac286f74"
SECRET_PASSAGE = "Passive sensors gather radiation that is emitted or reflected."
SAR_PASSAGE = "SAR instruments emit microwave pulses and record the echo."
STAC = "cdse_search_stac"


# ── fixtures ────────────────────────────────────────────────────────────────


def _private_doc(conversation: bool = False) -> dict:
    return {
        "id": "f6378a16-e113-4bbc-8ba8-7fd23b4b7e65",
        "version": 142,
        "score": 0.65,
        "reranking_score": 0.99,
        "collection_name": "Attached documents" if conversation else "RS",
        "payload": {
            "text": SECRET_PASSAGE,
            "metadata": {
                "source_name": "remote-sensing.pdf",
                "file_type": "application/pdf",
                "document_id": "6ab27b5c8bc8659543f544a2",
                "page": 4,
            },
            "user_id": OWNER_ID,
            "collection_id": "6ab27b3b8bc8659543f544a1",
        },
        "text": SECRET_PASSAGE,
        "metadata": {
            "source_name": "remote-sensing.pdf",
            "file_type": "application/pdf",
            "document_id": "6ab27b5c8bc8659543f544a2",
            "page": 4,
        },
    }


def _public_doc(text: str = SAR_PASSAGE, point: int = 481257, headers=("2. Sensors",)) -> dict:
    return {
        "id": point,
        "version": 7,
        "score": 0.71,
        "reranking_score": 0.88,
        "collection_name": "EVE open access",
        "payload": {
            "content": text,
            "title": "Active microwave remote sensing",
            "url": "https://doi.org/10.1000/example",
            "doi": "10.1000/example",
            "year": 2021,
            "journal": "Remote Sensing",
            "headers": list(headers),
        },
        "text": text,
        "metadata": {},
    }


def _wiley_doc() -> dict:
    return {
        "id": "10.1029/2020GL000001#1",
        "collection_name": "Wiley AI Gateway",
        "text": "Ocean heat content rose. " * 40,
        "metadata": {
            "additionalMetadata": {
                "title": "Paper 1",
                "journalTitle": "Geophysical Research Letters",
                "link": "https://doi.org/10.1029/2020GL000001",
                "publicationDate": "2021-03-01",
                "authors": ["A. Author"],
            }
        },
    }


def _stac_response() -> dict:
    item_id = "S2B_MSIL2A_20230904T091559_N0509_R093_T35TMF_20230904T120203"
    return {
        "catalog": "cdse",
        "search": {"collections": ["sentinel-2-l2a"], "limit": 1},
        "matched": 12,
        "more_available": True,
        "returned": 1,
        "items": [
            {
                "id": item_id,
                "collection": "sentinel-2-l2a",
                "datetime": "2023-09-04T09:15:59Z",
                "cloud_cover": 3,
                "platform": "sentinel-2b",
                "bbox": [26.0, 40.8, 27.3, 41.8],
                "self": f"https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items/{item_id}",
                "assets": {"thumbnail": {"href": "https://example.invalid/t.jpg"}},
            }
        ],
    }


def _retrieve_response(*docs: dict) -> dict:
    return {"retrieved_docs": list(docs), "latencies": {"reranking": 0.4}, "original_query": "q"}


def _tool_result(payload: Any, is_error: bool = False) -> CallToolResult:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=is_error)


def _request(name="retrieve", server="eve_retrieval", call_id="call-1"):
    return SimpleNamespace(
        name=name, server_name=server, args={}, runtime=SimpleNamespace(tool_call_id=call_id)
    )


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_INLINE_CITATIONS", True)


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_INLINE_CITATIONS", False)


@pytest.fixture
def cite_context():
    ctx, token = set_retrieval_context()
    ctx.cite = True
    try:
        yield ctx
    finally:
        reset_retrieval_context(token)


async def _intercept(payload, request=None, is_error=False):
    async def handler(_req):
        return _tool_result(payload, is_error=is_error)

    return await RetrievalContextInterceptor()(request or _request(), handler)


# ── descriptors ─────────────────────────────────────────────────────────────


class TestDescriptors:
    def test_public_passage_is_a_document_with_its_locator(self):
        source = source_from_document(extract_document_data(_public_doc()), rank=0, tool_call_id="call-1")
        assert source["kind"] == "document"
        assert source["provider"] == "qwen-512-filtered"
        assert source["provider_label"] == "EVE open access"
        assert source["visibility"] == "public"
        assert source["document_key"] == "doi:10.1000/example"
        assert source["locator"]["section"] == "2. Sensors"
        assert source["open"] == {"type": "url", "href": "https://doi.org/10.1000/example"}
        assert source["meta"]["venue"] == "Remote Sensing"
        assert source["tool_call_id"] == "call-1"
        assert re.fullmatch(r"src_[0-9a-f]{16}", source["id"])

    def test_id_is_stable_and_two_passages_of_one_document_differ(self):
        a1 = source_from_document(extract_document_data(_public_doc()), rank=0)
        a2 = source_from_document(extract_document_data(_public_doc()), rank=5)
        b = source_from_document(
            extract_document_data(_public_doc("Another passage.", point=9, headers=())), rank=1
        )
        assert a1["id"] == a2["id"]
        assert a1["document_key"] == b["document_key"]
        assert a1["id"] != b["id"]

    def test_private_file_is_a_user_file_without_owner_ids(self):
        source = source_from_document(extract_document_data(_private_doc()), rank=0)
        assert source["kind"] == "user_file"
        assert source["provider"] == "my-collections"
        assert source["visibility"] == "private"
        assert source["title"] == "remote-sensing.pdf"
        assert source["file"]["document_id"] == "6ab27b5c8bc8659543f544a2"
        assert source["file"]["collection_id"] == "6ab27b3b8bc8659543f544a1"
        assert source["open"]["type"] == "viewer"
        assert source["locator"]["page"] == 4
        public = json.dumps(public_source(source))
        assert OWNER_ID not in public
        assert "f6378a16" not in public  # the Qdrant point id stays server side

    def test_conversation_file_has_its_own_provider(self):
        source = source_from_document(extract_document_data(_private_doc(conversation=True)), rank=0)
        assert source["provider"] == "conversation"
        assert source["file"]["conversation_id"] == "6ab27b3b8bc8659543f544a1"

    def test_wiley_is_licensed_and_capped(self):
        source = source_from_document(extract_document_data(_wiley_doc()), rank=0)
        assert source["provider"] == "wiley"
        assert source["visibility"] == "licensed"
        assert source["document_key"] == "doi:10.1029/2020GL000001"
        assert len(source["passage"]) <= LICENSED_PASSAGE_MAX_CHARS
        assert source["passage"].endswith(".")

    def test_stac_item_has_a_readable_title_and_a_stac_link(self):
        item = _stac_response()["items"][0]
        source = source_from_stac_item(item, provider="cdse", rank=0, tool_call_id="call-2")
        assert source["kind"] == "stac_item"
        assert source["title"] == "Sentinel-2B L2A, tile 35TMF, 4 September 2023"
        assert source["open"]["type"] == "stac"
        assert source["locator"]["stac_item_id"] == item["id"]
        assert source["meta"] == {"cloud_cover": 3, "platform": "sentinel-2b"}
        assert "thumbnail" not in source  # never a third-party URL into an img

    def test_passage_cut_at_a_sentence(self):
        text = "First sentence here. " * 60
        cut = cut_passage(text, PASSAGE_MAX_CHARS)
        assert len(cut) <= PASSAGE_MAX_CHARS and cut.endswith(".")

    def test_record_without_text_is_not_a_source(self):
        doc = _public_doc(text="")
        doc["payload"]["content"] = ""
        assert source_from_document(extract_document_data(doc), rank=0) is None

    def test_retrieval_block_never_leaves(self):
        source = source_from_document(extract_document_data(_public_doc()), rank=3)
        assert source["retrieval"]["rank"] == 3
        assert "retrieval" not in public_source(source)


# ── rewriter ────────────────────────────────────────────────────────────────


def _registry(*sources: Dict[str, Any]) -> SourceRegistry:
    registry = SourceRegistry()
    for source in sources:
        registry.register(dict(source))
    return registry


def _rewrite(registry: SourceRegistry, tokens: List[str]):
    rewriter = CitationRewriter(registry)
    text = "".join(rewriter.feed(t) for t in tokens) + rewriter.flush()
    return text, rewriter


A = {"id": "src_a", "document_key": "doi:a", "title": "A", "retrieval": {"rank": 0}}
A2 = {"id": "src_a2", "document_key": "doi:a", "title": "A, second passage"}
B = {"id": "src_b", "document_key": "url:b", "title": "B"}


class TestRewriter:
    def test_no_citation_leaves_the_text(self):
        text, rewriter = _rewrite(_registry(A), ["SAR is ", "radar."])
        assert text == "SAR is radar."
        assert rewriter.citations() == []
        assert [s["id"] for s in rewriter.sources_retrieved()] == ["src_a"]

    def test_valid_citations_numbered_by_first_citation(self):
        text, rewriter = _rewrite(_registry(A, B), ["Optical [S2]. Radar [S", "1]. Again [S2]."])
        assert text == "Optical [1]. Radar [2]. Again [1]."
        cits = rewriter.citations()
        assert [(c["id"], c["n"]) for c in cits] == [("src_b", 1), ("src_a", 2)]
        assert cits[0]["at"] == [8, 30]
        assert marker_spans(text, cits) == ["[1]", "[1]", "[2]"]
        assert all("retrieval" not in c for c in cits)
        assert rewriter.sources_retrieved() == []

    def test_unknown_handle_is_stripped_and_counted(self):
        text, rewriter = _rewrite(_registry(A), ["Claim [S7]. Other [S1, S9]."])
        assert text == "Claim . Other [1]."
        assert rewriter.invalid == 2
        assert [c["id"] for c in rewriter.citations()] == ["src_a"]

    def test_two_passages_of_one_document_share_one_marker(self):
        text, rewriter = _rewrite(_registry(A, A2, B), ["Fact [S1][S2]. More [S3][S2]."])
        assert text == "Fact [1]. More [2][1]."
        cits = {c["id"]: c for c in rewriter.citations()}
        assert cits["src_a"]["n"] == cits["src_a2"]["n"] == 1
        assert cits["src_a"]["at"] == [5]
        assert cits["src_a2"]["at"] == [5, 18]

    def test_literal_brackets_and_split_markers(self):
        text, rewriter = _rewrite(_registry(A), ["x[1] and [", "S", "1", "] and [S", "1 is"])
        assert text == "x[1] and [1] and [S1 is"
        assert rewriter.citations()[0]["at"] == [9]

    def test_stream_ending_inside_a_marker_flushes_it_as_text(self):
        text, _ = _rewrite(_registry(A), ["end [S1"])
        assert text == "end [S1"


# ── interceptor ─────────────────────────────────────────────────────────────


class TestInterceptor:
    async def test_handles_reach_the_model_without_ids(self, flag_on, cite_context):
        result = await _intercept(_retrieve_response(_private_doc(), _public_doc()))
        slim = json.loads(result.content[0].text)
        assert [d["ref"] for d in slim["retrieved_docs"]] == ["S1", "S2"]
        assert slim["citation_instruction"] == CITATION_INSTRUCTION
        text = result.content[0].text
        assert not UUID_RE.search(text)
        assert OWNER_ID not in text and "481257" not in text
        sources = cite_context.sources.sources
        assert [s["handle"] for s in sources] == ["S1", "S2"]
        assert all(s["tool_call_id"] == "call-1" for s in sources)
        # The documents for the Sources panel are unchanged.
        assert len(cite_context.calls[0].documents) == 2

    async def test_flag_off_reduction_is_unchanged(self, flag_off, cite_context):
        response = _retrieve_response(_public_doc())
        result = await _intercept(response)
        assert json.loads(result.content[0].text) == slim_retrieval_response_for_model(response)
        assert "ref" not in result.content[0].text
        assert cite_context.sources.sources == []

    async def test_sync_run_gets_no_handles(self, flag_on):
        ctx, token = set_retrieval_context()  # cite stays False
        try:
            result = await _intercept(_retrieve_response(_public_doc()))
        finally:
            reset_retrieval_context(token)
        assert "ref" not in result.content[0].text
        assert ctx.sources.sources == []

    async def test_stac_items_get_handles(self, flag_on, cite_context):
        result = await _intercept(
            _stac_response(), request=_request(name="search_stac", server="cdse", call_id="call-2")
        )
        body = json.loads(result.content[0].text)
        assert body["items"][0]["ref"] == "S1"
        assert body["citation_instruction"] == CITATION_INSTRUCTION
        assert result.structuredContent is None
        [source] = cite_context.sources.sources
        assert source["kind"] == "stac_item" and source["provider"] == "cdse"
        assert cite_context.calls == []  # a STAC result is not a retrieval call

    async def test_tool_error_registers_nothing(self, flag_on, cite_context):
        await _intercept('{"error": "retrieval failed"}', is_error=True)
        await _intercept("Tool error: 502", request=_request(name="search_stac", server="cdse"), is_error=True)
        assert cite_context.sources.sources == []

    async def test_same_passage_twice_keeps_its_handle(self, flag_on, cite_context):
        await _intercept(_retrieve_response(_public_doc()))
        result = await _intercept(_retrieve_response(_public_doc(), _private_doc()), request=_request(call_id="call-9"))
        slim = json.loads(result.content[0].text)
        assert [d["ref"] for d in slim["retrieved_docs"]] == ["S1", "S2"]


# ── runner, stream, persistence, AG-UI ──────────────────────────────────────

ENDPOINT = {"requested": "main", "chain": ["main"], "answered": "main",
            "attempts": [], "circuit_open": [], "substituted": False}
T = "6a3f0c2e9b1d4e7f8a2b3c4d"
R = "0f8e6c1a-3b2d-4e5f-9a7b-1c2d3e4f5a6b"


def _resolver():
    return AsyncMock(return_value=(MagicMock(), {"endpoint": ENDPOINT}))


def _fill(at_index: int, payloads: List[tuple]):
    """``before_each`` hook: run the interceptor on each payload when the graph reaches ``at_index``."""

    def hook(index: int) -> None:
        if index != at_index:
            return
        ctx = get_retrieval_context()
        for payload, name, server, call_id in payloads:
            RetrievalContextInterceptor()._reduce(
                _tool_result(payload), f"{server}_{name}", tool_call_id=call_id, server_name=server
            )
        assert ctx is not None

    return hook


def _tokens(*parts: str):
    return [("messages", (_ai(p), {"langgraph_node": "agent"})) for p in parts]


def _ai(text):
    from langchain_core.messages import AIMessageChunk

    return AIMessageChunk(content=text)


async def _cited_run(answer_parts, payloads=None, cancel_event=None, before_each=None, **kw):
    payloads = payloads if payloads is not None else [
        (_retrieve_response(_public_doc(), _private_doc()), "retrieve", "eve_retrieval", "call-1")
    ]
    graph = _Graph(
        [_call(RETRIEVE, {"query": "SAR"}, "call-1"), _result(RETRIEVE, "call-1", content="{}"), *_tokens(*answer_parts)],
        before_each=before_each or _fill(1, payloads),
    )
    return await run_runner(
        graph,
        tools=[_tool(RETRIEVE, "eve_retrieval")],
        cancel_event=cancel_event,
        _resolve_agentic_llm_client=_resolver(),
        **kw,
    )


def _persist_kwargs(applied) -> Dict[str, Any]:
    return applied["persist_message_state"].await_args_list[-1].kwargs


class TestRunner:
    async def test_flag_off_frames_and_row_are_unchanged(self, flag_off):
        events, applied = await _cited_run(["SAR is radar [S1]."])
        final = _of(events, "final")[0]
        assert "citations" not in final and "sources_retrieved" not in final
        assert "".join(t["content"] for t in _of(events, "token")) == "SAR is radar [S1]."
        assert all(set(t) == {"type", "content"} for t in _of(events, "token"))
        kwargs = _persist_kwargs(applied)
        assert "citations" not in kwargs and "sources_retrieved" not in kwargs

    async def test_valid_citation_streams_persists_and_finals(self, flag_on):
        events, applied = await _cited_run(["SAR is radar [S", "1]. Files say so [S2]", "."])
        answer = "".join(t["content"] for t in _of(events, "token"))
        assert answer == "SAR is radar [1]. Files say so [2]."
        final = _of(events, "final")[0]
        assert final["answer"] == answer
        cits = final["citations"]
        assert [(c["n"], c["kind"]) for c in cits] == [(1, "document"), (2, "user_file")]
        assert marker_spans(answer, cits) == ["[1]", "[2]"]
        assert final["sources_retrieved"] == []
        # The token that closed a marker carries the list so far.
        tokens = _of(events, "token")
        closing = [t for t in tokens if "citations" in t]
        assert closing and len(closing[-1]["citations"]) == 2
        assert all("[" in t["content"] for t in closing)
        assert "citations" not in tokens[-1]  # the closing "." carries none
        kwargs = _persist_kwargs(applied)
        assert kwargs["output"] == answer
        assert kwargs["citations"] == cits
        assert kwargs["sources_retrieved"] == []
        assert OWNER_ID not in json.dumps(final)
        assert "retrieval" not in json.dumps(cits)

    async def test_no_citation_lists_every_source_as_retrieved(self, flag_on):
        events, applied = await _cited_run(["SAR is radar."])
        final = _of(events, "final")[0]
        assert final["answer"] == "SAR is radar."
        assert final["citations"] == []
        retrieved = final["sources_retrieved"]
        assert [s["provider"] for s in retrieved] == ["qwen-512-filtered", "my-collections"]
        assert all("passage" not in s for s in retrieved)
        assert _persist_kwargs(applied)["sources_retrieved"] == retrieved

    async def test_unknown_handle_is_stripped_from_stream_and_row(self, flag_on, caplog):
        caplog.set_level(logging.INFO)
        events, applied = await _cited_run(["Claim [S5]. Fact [S1]."])
        answer = "".join(t["content"] for t in _of(events, "token"))
        assert answer == "Claim . Fact [1]."
        assert _persist_kwargs(applied)["output"] == answer
        assert any("citations.summary" in r.getMessage() and "invalid=1" in r.getMessage() for r in caplog.records)

    async def test_two_passages_of_one_document_share_the_number(self, flag_on):
        payloads = [(
            _retrieve_response(_public_doc(), _public_doc("Another passage.", point=9, headers=())),
            "retrieve", "eve_retrieval", "call-1",
        )]
        events, _ = await _cited_run(["Both [S1][S2]."], payloads=payloads)
        final = _of(events, "final")[0]
        assert final["answer"] == "Both [1]."
        assert [c["n"] for c in final["citations"]] == [1, 1]
        assert {c["document_key"] for c in final["citations"]} == {"doi:10.1000/example"}

    async def test_private_file_citation(self, flag_on):
        payloads = [(_retrieve_response(_private_doc()), "retrieve", "eve_retrieval", "call-1")]
        events, _ = await _cited_run(["From your file [S1]."], payloads=payloads)
        [cit] = _of(events, "final")[0]["citations"]
        assert cit["kind"] == "user_file" and cit["visibility"] == "private"
        assert cit["file"]["name"] == "remote-sensing.pdf"

    async def test_stac_item_citation(self, flag_on):
        payloads = [(_stac_response(), "search_stac", "cdse", "call-1")]
        events, _ = await _cited_run(["A clear scene [S1]."], payloads=payloads)
        [cit] = _of(events, "final")[0]["citations"]
        assert cit["kind"] == "stac_item"
        assert cit["open"]["type"] == "stac"

    async def test_tool_error_leaves_no_sources(self, flag_on):
        payloads = [('{"error": "retrieval failed"}', "retrieve", "eve_retrieval", "call-1")]
        events, applied = await _cited_run(["No sources [S1]."], payloads=payloads)
        final = _of(events, "final")[0]
        assert final["answer"] == "No sources ."
        assert final["citations"] == [] and final["sources_retrieved"] == []

    async def test_stop_before_the_answer(self, flag_on):
        cancel = asyncio.Event()
        fill = _fill(1, [(_retrieve_response(_public_doc()), "retrieve", "eve_retrieval", "call-1")])

        def hook(index):
            fill(index)
            if index == 2:
                cancel.set()

        events, applied = await _cited_run(["Never [S1]."], cancel_event=cancel, before_each=hook)
        assert events[-1]["type"] == "stopped"
        assert _of(events, "token") == []
        kwargs = _persist_kwargs(applied)
        assert kwargs["stopped"] is True
        assert kwargs["output"] == ""
        assert kwargs["citations"] == []
        assert [s["provider"] for s in kwargs["sources_retrieved"]] == ["qwen-512-filtered"]

    async def test_no_user_text_in_logs(self, flag_on, caplog):
        caplog.set_level(logging.DEBUG)
        await _cited_run(["Secret answer sentence [S1] [S9]."], query="my private question")
        logged = "\n".join(r.getMessage() for r in caplog.records)
        for text in ("Secret answer sentence", "my private question", SECRET_PASSAGE, SAR_PASSAGE, "remote-sensing.pdf"):
            assert text not in logged

    async def test_agui_transcript_carries_citations_and_validates(self, flag_on):
        events, _ = await _cited_run(["SAR is radar [S1]", "."])
        adapter = AguiAdapter(RunContext(thread_id=T, run_id=R, message_id="m1", trace_id="a" * 32, graph="react"))
        out: List[Any] = list(adapter.start())
        for frame in events:
            out.extend(adapter.adapt(frame))
        out.extend(adapter.finish())
        body = encode_sse(out)
        assert validate_transcript(body, thread_id=T, run_id=R) == []
        parsed = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        [end] = [e for e in parsed if e["type"] == "TEXT_MESSAGE_END"]
        assert [c["n"] for c in end["metadata"]["eve.citations"]] == [1]
        assert [s["provider"] for s in end["metadata"]["eve.sourcesRetrieved"]] == ["my-collections"]
        content = [e for e in parsed if e["type"] == "TEXT_MESSAGE_CONTENT"]
        assert "metadata" in content[0] and "metadata" not in content[1]

    async def test_agui_flag_off_has_no_citation_metadata(self, flag_off):
        events, _ = await _cited_run(["SAR is radar [S1]."])
        adapter = AguiAdapter(RunContext(thread_id=T, run_id=R, message_id="m1", trace_id="a" * 32, graph="react"))
        out: List[Any] = list(adapter.start())
        for frame in events:
            out.extend(adapter.adapt(frame))
        body = encode_sse(out)
        assert "eve.citations" not in body and "eve.sourcesRetrieved" not in body


class TestPersist:
    async def test_citations_land_in_metadata(self, monkeypatch):
        from src.services import generate_answer as ga

        message = SimpleNamespace(metadata={"latencies": {"x": 1}}, save=AsyncMock())
        monkeypatch.setattr(ga.Message, "find_by_id_on_primary", AsyncMock(return_value=message))
        await ga.persist_message_state("m1", citations=[{"id": "src_a", "n": 1, "at": [0]}], sources_retrieved=[])
        assert message.metadata == {
            "latencies": {"x": 1},
            "citations": [{"id": "src_a", "n": 1, "at": [0]}],
            "sources_retrieved": [],
        }
