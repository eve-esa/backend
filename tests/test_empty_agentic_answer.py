"""An agentic turn that ends normally with nothing to show is a failed turn.

Before: the row kept ``output: ""`` with no ``metadata.error``, the stream
ended with an empty ``final``, nothing was logged above INFO and the root span
looked healthy, while the UI showed the generic failure copy. Now the turn is
persisted with ``metadata.error.code == "empty_answer"``, the stream ends with
the ``error`` frame the frontend renders with Retry, one WARNING line names the
ids and the root span carries the error.
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from src import observability
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import generate_answer_agentic
from src.services.generate_answer import build_empty_answer_payload
from src.services.mcp.artifact_context import get_artifact_context
from tests.test_agent_step_frames import _Graph, _answer, _call, _of, _result, _run
from tests.test_agentic_fallback import _FakeUpdatesGraph, _patched_runner
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

SECRET_QUERY = "my-private-question-text"
UNPARSED_CALL = '[TOOL_CALLS]eve_retrieval_retrieve{"query": "sea ice'
PARSED_CALL = '[TOOL_CALLS]eve_retrieval_retrieve{"query": "sea ice"}'
EMPTY_LINE = "generation.empty_answer conversation_id=c1 message_id=m1"
RECOVERED = "Primary agent node failed after agent_fallback produced an answer"
EMPTY_DETAIL = {
    "code": "empty_answer",
    "message": build_empty_answer_payload()["message"],
}


@pytest.fixture
def telemetry(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = observability.build_tracer_provider(exporter, batch=False)
    monkeypatch.setitem(observability._state, "tracer_provider", provider)
    yield exporter
    provider.shutdown()


def _persisted(applied):
    return applied["persist_message_state"].await_args.kwargs


def _root(exporter):
    [root] = [s for s in exporter.get_finished_spans() if s.name == "invoke_agent"]
    return root


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


# ─── stream ──────────────────────────────────────────────────────────────────


@pytest.mark.no_db
class TestStreamEmptyAnswer:
    @pytest.mark.parametrize(
        "events",
        [
            [],
            [_answer("  \n\t ")],
            [_answer(UNPARSED_CALL)],
            [_answer(PARSED_CALL)],
        ],
        ids=["nothing", "whitespace", "unparsed-tool-call-text", "unrun-tool-call-text"],
    )
    async def test_ends_with_the_empty_answer_error(self, events):
        frames, applied = await _run(_Graph(events), query=SECRET_QUERY)

        assert frames[-1] == {
            "type": "error",
            "code": "empty_answer",
            "message": build_empty_answer_payload()["message"],
        }
        assert _of(frames, "final") == []
        kwargs = _persisted(applied)
        assert kwargs["error"] == build_empty_answer_payload()
        assert kwargs["output"] == ""
        assert not kwargs.get("clear_error")
        assert "[TOOL_CALLS]" not in json.dumps(frames)

    async def test_logs_one_warning_with_the_ids_and_no_text(self, caplog):
        caplog.set_level(logging.INFO)
        await _run(_Graph([_answer(UNPARSED_CALL)]), query=SECRET_QUERY)

        warnings = _warnings(caplog)
        assert warnings.count(EMPTY_LINE) == 1
        assert not any(SECRET_QUERY in w or "sea ice" in w for w in warnings)

    async def test_root_span_carries_the_error_code(self, telemetry):
        await _run(_Graph([_answer("   ")]))

        root = _root(telemetry)
        assert root.status.status_code == StatusCode.ERROR
        assert root.attributes["eve.error.code"] == "empty_answer"

    async def test_after_a_tool_result_is_still_an_empty_answer(self):
        frames, applied = await _run(
            _Graph(
                [
                    _call("eve_retrieval_retrieve", {"query": "q"}, "call-1"),
                    _result("eve_retrieval_retrieve", "call-1"),
                    _answer(UNPARSED_CALL),
                ]
            )
        )

        assert frames[-1]["type"] == "error"
        assert frames[-1]["code"] == "empty_answer"
        assert _persisted(applied)["error"]["code"] == "empty_answer"


def _fallback_update():
    return ("updates", {"agent_fallback": {"messages": [AIMessage(content="")]}})


def _fallback_answer(text, message_id=None):
    return (
        "messages",
        (AIMessage(content=text, id=message_id), {"langgraph_node": "agent_fallback"}),
    )


def _recovery_descriptions(applied):
    return [
        call.kwargs.get("description")
        for call in applied["persist_policy_event"].await_args_list
    ]


@pytest.mark.no_db
class TestStreamEdges:
    async def test_a_stop_after_the_last_event_leaves_the_root_span_unmarked(self, telemetry):
        # The Stop lands once the graph is done: the event is set, then the
        # task is cancelled at the next await. The turn ends stopped, so the
        # trace must not read empty_answer.
        stop = asyncio.Event()

        class _StopAtTheEnd(_Graph):
            async def astream(self, *args, **kwargs):
                async for event in super().astream(*args, **kwargs):
                    yield event
                stop.set()

        frames, applied = await _run(
            _StopAtTheEnd([_answer("   ")]),
            cancel_event=stop,
            append_missing_artifact_stubs=AsyncMock(side_effect=asyncio.CancelledError),
        )

        assert _of(frames, "error") == []
        assert _persisted(applied)["stopped"] is True
        root = _root(telemetry)
        assert root.status.status_code != StatusCode.ERROR
        assert "eve.error.code" not in root.attributes

    async def test_whitespace_from_the_fallback_does_not_recover_a_failed_primary(self):
        frames, applied = await _run(
            _Graph(
                # The second model call flushes the first one's whitespace
                # onto the stream before the primary's failure lands.
                [
                    _fallback_update(),
                    _fallback_answer("  \n", "fb-1"),
                    _fallback_answer(" ", "fb-2"),
                ],
                raise_exc=TimeoutError("agent timeout"),
            )
        )

        # The primary's timeout stands. The whitespace already went out, so
        # the failure-after-text rule carries it on `final`.
        assert "empty_answer" not in json.dumps(frames)
        assert frames[-1]["error"]["code"] == "timeout"
        assert _persisted(applied)["error"]["code"] == "timeout"
        assert RECOVERED not in _recovery_descriptions(applied)


@pytest.mark.no_db
class TestStreamUnchanged:
    async def test_an_answer_ends_with_final_and_no_error(self, caplog, telemetry):
        caplog.set_level(logging.INFO)
        frames, applied = await _run(_Graph([_answer("Sea ice is frozen sea water.")]))

        assert frames[-1]["type"] == "final"
        assert "error" not in frames[-1]
        assert _persisted(applied).get("error") is None
        assert _persisted(applied)["clear_error"] is True
        assert not any("generation.empty_answer" in w for w in _warnings(caplog))
        assert _root(telemetry).status.status_code != StatusCode.ERROR
        assert "eve.error.code" not in _root(telemetry).attributes

    async def test_a_stop_before_any_text_stays_stopped(self, caplog):
        import asyncio

        caplog.set_level(logging.INFO)
        stop = asyncio.Event()
        graph = _Graph(
            [_call("eve_retrieval_retrieve", {"query": "q"}, "call-1"), _answer("late")],
            before_each=lambda index: stop.set() if index == 1 else None,
        )

        frames, applied = await _run(graph, cancel_event=stop)

        assert frames[-1]["type"] == "stopped"
        assert _of(frames, "error") == []
        kwargs = _persisted(applied)
        assert kwargs["stopped"] is True
        assert kwargs.get("error") is None
        assert not any("generation.empty_answer" in w for w in _warnings(caplog))

    async def test_a_failed_turn_keeps_its_own_error(self, caplog):
        caplog.set_level(logging.INFO)
        frames, applied = await _run(_Graph([], raise_exc=RuntimeError("boom")))

        assert frames[-1]["type"] == "error"
        assert frames[-1]["code"] == "upstream_error"
        assert _persisted(applied)["error"]["code"] == "upstream_error"
        assert not any("generation.empty_answer" in w for w in _warnings(caplog))

    async def test_artifacts_without_text_are_not_an_error(self, caplog):
        caplog.set_level(logging.INFO)

        def _tool_saved_an_artifact(index):
            if index == 1:
                get_artifact_context().collected_artifact_ids.append("art-1")

        frames, applied = await _run(
            _Graph(
                [_call("make_plot", {}, "call-1"), _result("make_plot", "call-1")],
                before_each=_tool_saved_an_artifact,
            )
        )

        assert frames[-1]["type"] == "final"
        assert frames[-1]["artifact_ids"] == ["art-1"]
        kwargs = _persisted(applied)
        assert kwargs.get("error") is None
        assert kwargs["artifact_ids"] == ["art-1"]
        assert not any("generation.empty_answer" in w for w in _warnings(caplog))


# ─── sync runner ─────────────────────────────────────────────────────────────


async def _sync(updates, telemetry=None):
    request = GenerationRequest(query=SECRET_QUERY, llm_type="main", agent="react")
    with _patched_runner(_build_react_graph=MagicMock(return_value=_FakeUpdatesGraph(updates))):
        return await generate_answer_agentic(
            request, user_id="u1", conversation_id="c1", message_id="m1"
        )


@pytest.mark.no_db
class TestSyncRunner:
    @pytest.mark.parametrize("text", ["", "   ", UNPARSED_CALL, PARSED_CALL])
    async def test_empty_final_logs_marks_and_returns_no_text(self, text, caplog, telemetry):
        caplog.set_level(logging.INFO)
        result = await _sync([{"agent": {"messages": [AIMessage(content=text)]}}])

        assert result[0].strip() == ""
        assert "[TOOL_CALLS]" not in result[0]
        warnings = _warnings(caplog)
        assert warnings.count(EMPTY_LINE) == 1
        assert not any(SECRET_QUERY in w for w in warnings)
        root = _root(telemetry)
        assert root.status.status_code == StatusCode.ERROR
        assert root.attributes["eve.error.code"] == "empty_answer"

    async def test_tool_call_text_before_an_answer_keeps_the_answer(self, caplog):
        caplog.set_level(logging.INFO)
        result = await _sync(
            [{"agent": {"messages": [AIMessage(content=PARSED_CALL + "Sea ice melts.")]}}]
        )

        assert result[0] == "Sea ice melts."
        assert not any("generation.empty_answer" in w for w in _warnings(caplog))

    async def test_list_content_is_joined_as_text(self, caplog):
        caplog.set_level(logging.INFO)
        content = [{"type": "text", "text": "Sea ice "}, {"type": "text", "text": "melts."}]
        result = await _sync([{"agent": {"messages": [AIMessage(content=content)]}}])

        assert result[0] == "Sea ice melts."
        assert not any("generation.empty_answer" in w for w in _warnings(caplog))

    async def test_list_content_with_blank_text_is_an_empty_answer(self, caplog):
        caplog.set_level(logging.INFO)
        content = [{"type": "text", "text": "  "}]
        result = await _sync([{"agent": {"messages": [AIMessage(content=content)]}}])

        assert result[0] == ""
        assert _warnings(caplog).count(EMPTY_LINE) == 1

    async def test_whitespace_from_the_fallback_does_not_recover_a_failed_primary(self):
        updates = [
            {"agent_fallback": {"messages": [AIMessage(content="")]}},
            {"agent_fallback": {"messages": [AIMessage(content="  ")]}},
        ]
        graph = _FakeUpdatesGraph(updates, raise_exc=TimeoutError("agent timeout"))
        request = GenerationRequest(query=SECRET_QUERY, llm_type="main", agent="react")

        with _patched_runner(_build_react_graph=MagicMock(return_value=graph)) as patched:
            with pytest.raises(TimeoutError):
                await generate_answer_agentic(
                    request, user_id="u1", conversation_id="c1", message_id="m1"
                )

        assert RECOVERED not in _recovery_descriptions(patched)

    async def test_an_answer_is_untouched(self, caplog, telemetry):
        caplog.set_level(logging.INFO)
        result = await _sync([{"agent": {"messages": [AIMessage(content="An answer.")]}}])

        assert result[0] == "An answer."
        assert not any("generation.empty_answer" in w for w in _warnings(caplog))
        assert _root(telemetry).status.status_code != StatusCode.ERROR


# ─── sync routes ─────────────────────────────────────────────────────────────


def _agentic_result(answer, artifact_ids=()):
    # (answer, documents, use_rag, latencies, prompts, trace, artifact_ids)
    return answer, [], False, {}, {}, [], list(artifact_ids)


async def _post_generate_agentic(async_client, monkeypatch, result):
    monkeypatch.setattr(
        "src.routers.message.generate_answer_agentic", AsyncMock(return_value=result)
    )
    user, token = await create_test_user_and_token()
    conversation = Conversation(user_id=user.id, name="empty-answer")
    await conversation.save()
    response = await async_client.post(
        f"/conversations/{conversation.id}/generate-agentic",
        json={"query": SECRET_QUERY},
        headers={"Authorization": f"Bearer {token}"},
    )
    [saved] = await Message.find_all(filter_dict={"conversation_id": conversation.id})
    return user, conversation, saved, response


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["", "  \n "])
async def test_generate_agentic_persists_an_empty_answer_error(async_client, monkeypatch, answer):
    user, conversation, saved, response = await _post_generate_agentic(
        async_client, monkeypatch, _agentic_result(answer)
    )
    try:
        # An API client tells it from a success: same status and detail shape
        # as any other failed generation, and the row keeps the marker.
        assert response.status_code == 500, response.text
        assert response.json()["detail"] == EMPTY_DETAIL
        assert saved.output == ""
        assert saved.metadata["error"] == build_empty_answer_payload()
        assert saved.metadata["pipeline"] == "agentic"
    finally:
        await cleanup_models([saved, conversation, user])


@pytest.mark.asyncio
async def test_generate_agentic_with_artifacts_and_no_text_is_not_an_error(
    async_client, monkeypatch
):
    user, conversation, saved, response = await _post_generate_agentic(
        async_client, monkeypatch, _agentic_result("", ["art-1"])
    )
    try:
        assert response.status_code == 200, response.text
        assert "error" not in saved.metadata
        assert saved.artifact_ids == ["art-1"]
    finally:
        await cleanup_models([saved, conversation, user])


@pytest.mark.asyncio
async def test_generate_agentic_answer_has_no_error(async_client, monkeypatch):
    user, conversation, saved, response = await _post_generate_agentic(
        async_client, monkeypatch, _agentic_result("An answer.")
    )
    try:
        assert response.status_code == 200, response.text
        assert saved.output == "An answer."
        assert "error" not in saved.metadata
    finally:
        await cleanup_models([saved, conversation, user])


async def _retry_agentic(async_client, monkeypatch, result):
    from tests.test_message_retry import _make_conversation_and_message

    monkeypatch.setattr(
        "src.routers.message.generate_answer_agentic", AsyncMock(return_value=result)
    )
    user, token = await create_test_user_and_token()
    conversation, message = await _make_conversation_and_message(user, agentic=True)
    response = await async_client.post(
        f"/conversations/{conversation.id}/messages/{message.id}/retry",
        headers={"Authorization": f"Bearer {token}"},
    )
    saved = await Message.find_by_id(message.id)
    return user, conversation, saved, response


@pytest.mark.asyncio
async def test_agentic_retry_with_whitespace_is_an_empty_answer(async_client, monkeypatch):
    user, conversation, saved, response = await _retry_agentic(
        async_client, monkeypatch, _agentic_result(" \n")
    )
    try:
        assert response.status_code == 500, response.text
        assert response.json()["detail"] == EMPTY_DETAIL
        assert saved.output == ""
        assert saved.metadata["error"] == build_empty_answer_payload()
        assert saved.stopped is False
    finally:
        await cleanup_models([saved, conversation, user])


@pytest.mark.asyncio
async def test_agentic_retry_with_artifacts_and_no_text_clears_the_error(
    async_client, monkeypatch
):
    user, conversation, saved, response = await _retry_agentic(
        async_client, monkeypatch, _agentic_result("", ["art-1"])
    )
    try:
        assert response.status_code == 200, response.text
        assert "error" not in saved.metadata
    finally:
        await cleanup_models([saved, conversation, user])
