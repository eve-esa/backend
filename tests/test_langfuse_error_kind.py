"""Langfuse events for error_logs.kind (filterable timeout / retry / tool_error)."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.error_logger import ErrorLogger, set_conversation_context
from src.utils.langfuse_helper import (
    _callback_handler_ctx,
    _node_spans_by_session,
    _root_span_ctx,
    _trace_by_session,
    record_error_kind,
)

pytestmark = pytest.mark.no_db

_TRACE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_SPAN = "bbbbbbbbbbbbbbbb"


@pytest.fixture(autouse=True)
def _clear_langfuse_span_cache():
    _node_spans_by_session.clear()
    yield
    _node_spans_by_session.clear()


def _enabled_client():
    client = MagicMock()
    client.get_current_trace_id.return_value = None
    client.get_current_observation_id.return_value = None
    return client


class TestRecordErrorKind:
    def test_noop_when_disabled(self):
        with patch("src.utils.langfuse_helper.is_langfuse_enabled", return_value=False):
            record_error_kind("timeout", node="agent", graph="react")

    def test_timeout_event_nests_under_named_node_span(self):
        client = _enabled_client()
        root = MagicMock()
        agent = MagicMock()
        agent._otel_span = MagicMock()
        agent._otel_span.name = "agent"
        handler = MagicMock()
        handler._runs = {"run-1": agent}
        token_r = _root_span_ctx.set(root)
        token_h = _callback_handler_ctx.set(handler)
        try:
            with (
                patch(
                    "src.utils.langfuse_helper.is_langfuse_enabled", return_value=True
                ),
                patch(
                    "src.utils.langfuse_helper._get_langfuse_client",
                    return_value=client,
                ),
                patch("src.utils.langfuse_helper._ensure_langfuse_host"),
            ):
                record_error_kind("timeout", node="agent", graph="react")
        finally:
            _root_span_ctx.reset(token_r)
            _callback_handler_ctx.reset(token_h)

        agent.create_event.assert_called_once()
        assert agent.create_event.call_args.kwargs["name"] == "timeout"
        root.create_event.assert_not_called()

    def test_retry_parents_the_previous_agent_span(self):
        client = _enabled_client()
        first = MagicMock()
        first._otel_span = MagicMock()
        first._otel_span.name = "agent"
        second = MagicMock()
        second._otel_span = MagicMock()
        second._otel_span.name = "agent"
        _node_spans_by_session["_default"] = {"agent": [first, second]}
        with (
            patch("src.utils.langfuse_helper.is_langfuse_enabled", return_value=True),
            patch("src.utils.langfuse_helper._get_langfuse_client", return_value=client),
            patch("src.utils.langfuse_helper._ensure_langfuse_host"),
        ):
            record_error_kind("retry", node="agent", extra={"attempt": 2})
        first.create_event.assert_called_once()
        second.create_event.assert_not_called()
        assert first.create_event.call_args.kwargs["name"] == "retry"

    def test_timeout_uses_cached_span_after_handler_pops_runs(self):
        client = _enabled_client()
        agent = MagicMock()
        agent._otel_span = MagicMock()
        agent._otel_span.name = "agent"
        _node_spans_by_session["_default"] = {"agent": [agent]}
        handler = MagicMock()
        handler._runs = {}
        token_h = _callback_handler_ctx.set(handler)
        try:
            with (
                patch(
                    "src.utils.langfuse_helper.is_langfuse_enabled", return_value=True
                ),
                patch(
                    "src.utils.langfuse_helper._get_langfuse_client",
                    return_value=client,
                ),
                patch("src.utils.langfuse_helper._ensure_langfuse_host"),
            ):
                record_error_kind("timeout", node="agent")
                record_error_kind("fallback", node="agent")
        finally:
            _callback_handler_ctx.reset(token_h)
        names = [c.kwargs["name"] for c in agent.create_event.call_args_list]
        assert names == ["timeout", "fallback"]

    def test_timeout_event_uses_root_span_not_as_root(self):
        client = _enabled_client()
        root = MagicMock()
        token = _root_span_ctx.set(root)
        try:
            with (
                patch(
                    "src.utils.langfuse_helper.is_langfuse_enabled", return_value=True
                ),
                patch(
                    "src.utils.langfuse_helper._get_langfuse_client",
                    return_value=client,
                ),
                patch("src.utils.langfuse_helper._ensure_langfuse_host"),
            ):
                record_error_kind(
                    "timeout",
                    node="agent",
                    graph="react",
                    description="Policy 'timeout' on node 'agent'",
                )
        finally:
            _root_span_ctx.reset(token)

        root.create_event.assert_called_once()
        kwargs = root.create_event.call_args.kwargs
        assert kwargs["name"] == "timeout"
        assert kwargs["level"] == "ERROR"
        assert kwargs["metadata"]["kind"] == "timeout"
        assert "kind:timeout" in kwargs["metadata"]["tags"]
        assert "trace_context" not in kwargs
        client.create_event.assert_not_called()
        client.flush.assert_called()

    def test_retry_is_warning(self):
        client = _enabled_client()
        root = MagicMock()
        token = _root_span_ctx.set(root)
        try:
            with (
                patch(
                    "src.utils.langfuse_helper.is_langfuse_enabled", return_value=True
                ),
                patch(
                    "src.utils.langfuse_helper._get_langfuse_client",
                    return_value=client,
                ),
                patch("src.utils.langfuse_helper._ensure_langfuse_host"),
            ):
                record_error_kind("retry", node="agent", extra={"attempt": 2})
        finally:
            _root_span_ctx.reset(token)

        kwargs = root.create_event.call_args.kwargs
        assert kwargs["name"] == "retry"
        assert kwargs["level"] == "WARNING"
        assert kwargs["metadata"]["attempt"] == 2

    def test_tool_error_keeps_mcp_signal(self):
        client = _enabled_client()
        root = MagicMock()
        token = _root_span_ctx.set(root)
        try:
            with (
                patch(
                    "src.utils.langfuse_helper.is_langfuse_enabled", return_value=True
                ),
                patch(
                    "src.utils.langfuse_helper._get_langfuse_client",
                    return_value=client,
                ),
                patch("src.utils.langfuse_helper._ensure_langfuse_host"),
            ):
                record_error_kind(
                    "tool_error",
                    node="tools",
                    source="mcp",
                    extra={"signal": "text", "tool": "dummy_fail", "server": "dummy"},
                )
        finally:
            _root_span_ctx.reset(token)

        meta = root.create_event.call_args.kwargs["metadata"]
        assert meta["kind"] == "tool_error"
        assert meta["signal"] == "text"

    def test_session_map_parents_without_trace_context_flag(self):
        client = _enabled_client()
        conv = "conversation-after-exit"
        _trace_by_session[conv] = (_TRACE, _SPAN)
        set_conversation_context(conv)
        try:
            with (
                patch(
                    "src.utils.langfuse_helper.is_langfuse_enabled", return_value=True
                ),
                patch(
                    "src.utils.langfuse_helper._get_langfuse_client",
                    return_value=client,
                ),
                patch("src.utils.langfuse_helper._ensure_langfuse_host"),
            ):
                record_error_kind("timeout", node="agent", source="runner")
        finally:
            set_conversation_context(None)
            _trace_by_session.pop(conv, None)

        client.create_event.assert_called_once()
        kwargs = client.create_event.call_args.kwargs
        assert kwargs["name"] == "timeout"
        assert "trace_context" not in kwargs

    def test_no_parent_does_not_open_a_root_trace(self):
        client = _enabled_client()
        with (
            patch("src.utils.langfuse_helper.is_langfuse_enabled", return_value=True),
            patch("src.utils.langfuse_helper._get_langfuse_client", return_value=client),
            patch("src.utils.langfuse_helper._ensure_langfuse_host"),
        ):
            record_error_kind("timeout", node="agent")
        client.create_event.assert_not_called()

    def test_sdk_failure_does_not_raise(self):
        client = _enabled_client()
        root = MagicMock()
        root.create_event.side_effect = RuntimeError("langfuse down")
        token = _root_span_ctx.set(root)
        try:
            with (
                patch(
                    "src.utils.langfuse_helper.is_langfuse_enabled", return_value=True
                ),
                patch(
                    "src.utils.langfuse_helper._get_langfuse_client",
                    return_value=client,
                ),
                patch("src.utils.langfuse_helper._ensure_langfuse_host"),
            ):
                record_error_kind("timeout", node="agent")
        finally:
            _root_span_ctx.reset(token)


class TestLogErrorMirrorsKindToLangfuse:
    @pytest.mark.asyncio
    async def test_log_error_records_kind(self):
        logger = ErrorLogger()
        with patch("src.utils.error_logger.record_error_kind") as mirrored:
            await logger.log_error(
                RuntimeError("idle"),
                description="Policy 'timeout' on node 'agent'",
                kind="timeout",
                node="agent",
                graph="react",
            )
        mirrored.assert_called_once()
        assert len(logger._buffer) == 1
        assert logger._buffer[0].kind == "timeout"
        if logger._flush_task and not logger._flush_task.done():
            logger._flush_task.cancel()
        logger._buffer.clear()

    @pytest.mark.asyncio
    async def test_mcp_load_does_not_hit_langfuse(self):
        logger = ErrorLogger()
        with patch("src.utils.error_logger.record_error_kind") as mirrored:
            await logger.log_error(
                RuntimeError("down"),
                description="server skipped",
                kind="mcp_load",
                source="mcp_load",
            )
        mirrored.assert_not_called()
        assert len(logger._buffer) == 1
        if logger._flush_task and not logger._flush_task.done():
            logger._flush_task.cancel()
        logger._buffer.clear()

    @pytest.mark.asyncio
    async def test_rag_still_writes_mongo(self):
        logger = ErrorLogger()
        with patch("src.utils.error_logger.record_error_kind"):
            await logger.log_error(
                RuntimeError("llm"),
                description="generation failed",
                kind="rag",
                component="LLM",
                pipeline_stage="generation",
            )
        assert len(logger._buffer) == 1
        assert logger._buffer[0].kind == "rag"
        if logger._flush_task and not logger._flush_task.done():
            logger._flush_task.cancel()
        logger._buffer.clear()
