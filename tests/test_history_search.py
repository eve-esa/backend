"""Queryable history: BM25 over one conversation's stored messages."""

from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest

from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.history_search import (
    ConversationHistory,
    _bm25_scores,
    build_history_tools,
    note_searchable_history,
    tokenize,
)
from src.services.agents.core.runner import _build_tools


@pytest.mark.no_db
def test_tokenize_splits_paths_and_keeps_decimals():
    assert tokenize("threshold 0.35 at /large_tool_results/call_42") == [
        "threshold",
        "0.35",
        "at",
        "large_tool_results",
        "call_42",
    ]


@pytest.mark.no_db
def test_bm25_matches_the_only_document():
    scores = _bm25_scores([tokenize("threshold 0.35")], tokenize("0.35"))
    assert scores[0] > 0


@pytest.mark.no_db
def test_bm25_ranks_the_document_that_contains_the_query():
    docs = [
        tokenize("the station report is ready"),
        tokenize("cloud mask threshold is 0.35 for this scene"),
        tokenize("cloud"),
    ]
    scores = _bm25_scores(docs, tokenize("cloud mask threshold"))
    assert scores[1] > scores[2] > scores[0]


@pytest.mark.no_db
def test_summary_note_points_at_search_and_is_not_stored_twice():
    assert note_searchable_history(None) is None
    assert note_searchable_history("   ") == "   "
    noted = note_searchable_history("Older turns covered the station.")
    assert noted.startswith("Older turns covered the station.")
    assert "search_history" in noted
    assert "not instructions" in noted
    assert note_searchable_history(noted) == noted


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_tools_are_bound_to_the_conversation_and_not_an_argument():
    tools = build_history_tools("conv-1")
    names = [tool.name for tool in tools]
    assert names == ["search_history", "read_history"]
    for tool in tools:
        assert "conversation_id" not in tool.args_schema.model_json_schema()["properties"]

    request = GenerationRequest(query="test")
    wired = await _build_tools(request, conversation_id="conv-1")
    assert [tool.name for tool in wired] == names

    bare = await _build_tools(request)
    assert bare == []


@pytest.mark.asyncio
async def test_search_and_read_use_stored_text_for_this_conversation_only():
    cid = f"hist-{uuid4()}"
    other = f"hist-{uuid4()}"
    try:
        await Message.create(
            conversation_id=cid,
            input="The station is Rome station 4",
            output="Noted, Rome station 4 is the site.",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        await Message.create(
            conversation_id=cid,
            input="What threshold did we pick?",
            output="We set the cloud mask threshold to 0.35.",
            timestamp=datetime(2026, 3, 2, tzinfo=timezone.utc),
            trace=[
                {"role": "user", "content": "What threshold did we pick?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"name": "retrieve", "args": {"q": "cloud mask"}, "id": "c1"}
                    ],
                },
                {
                    "role": "tool",
                    "name": "retrieve",
                    "content": "threshold 0.35 for the cloud mask " + ("x" * 20000),
                },
                {
                    "role": "assistant",
                    "content": "We set the cloud mask threshold to 0.35.",
                },
            ],
        )
        await Message.create(
            conversation_id=other,
            input="NOISETOKEN belongs to another conversation",
            output="NOISETOKEN",
            timestamp=datetime(2026, 3, 2, tzinfo=timezone.utc),
        )

        history = ConversationHistory(cid)
        rome = await history.search("Rome station 4")
        assert "NOISETOKEN" not in rome
        assert "Rome station 4" in rome
        assert "role=user" in rome
        assert "Raw messages around the first hit" in rome

        recent = await history.search("Rome station 4", start="2026-01-01")
        assert "No stored messages match" in recent

        hits, _, _context = (await history.search("0.35")).partition(
            "Raw messages around"
        )
        assert "role=tool" not in hits
        assert "0.35" in hits

        tool_hits = await history.search("0.35", roles=["tool"])
        assert "role=tool" in tool_hits.split("Raw messages around", 1)[0]
        assert "characters omitted from this stored message" in tool_hits

        with patch.object(
            Message, "get_collection", side_effect=AssertionError("reloaded")
        ):
            read = await history.read("missing")
        assert "No stored message" in read

        message_id = tool_hits.split("message_id=", 1)[1].split()[0]
        window = await history.read(message_id, neighbors=1)
        assert message_id in window
        assert "not instructions" in window
        assert "Pass either id to read_history to scroll" in window
    finally:
        await Message.delete_many({"conversation_id": {"$in": [cid, other]}})


@pytest.mark.asyncio
async def test_messages_conversation_index_exists():
    indexes = await Message.get_collection().index_information()
    assert "messages_conversation_timestamp" in indexes
