"""Unit tests for agentic error-log taxonomy (policy + tool_error)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.agents.core.interceptors import (
    ErrorLoggingInterceptor,
    ToolReturnedError,
    classify_mcp_tool_result,
)
from src.services.agents.graphs_bundle import graphs_base_module
from src.utils.error_logger import ErrorLogger, PolicyEvent, persist_policy_event

pytestmark = pytest.mark.no_db


class TestErrorLoggerAgenticFields:
    def test_agentic_fields_are_not_mirrored_into_legacy_columns(self):
        logger = ErrorLogger()
        doc = logger._create_error_document(
            error=RuntimeError("boom"),
            description="retry fired",
            kind="policy",
            policy="retry",
            node="agent",
            graph="react",
        )
        assert doc.kind == "retry"
        assert doc.policy is None
        assert doc.node == "agent"
        assert doc.graph == "react"
        assert doc.component is None
        assert doc.pipeline_stage is None
        assert doc.error["type"] == "RuntimeError"
        assert doc.error["message"] == "boom"
        assert "args" not in doc.error
        assert doc.error_type is None
        assert "attempt" not in doc.error
        dumped = doc.to_dict()
        assert dumped["kind"] == "retry"
        assert "policy" not in dumped
        assert "component" not in dumped
        assert "pipeline_stage" not in dumped
        assert "error_type" not in dumped

    def test_source_is_not_copied_into_component(self):
        logger = ErrorLogger()
        doc = logger._create_error_document(
            error=PolicyEvent("skipped"),
            description="mcp skip",
            kind="policy",
            policy="mcp_load",
            source="mcp_load",
        )
        assert doc.component is None
        assert doc.source == "mcp_load"
        assert doc.pipeline_stage is None

    def test_legacy_component_stage_still_win_when_passed(self):
        from src.utils.error_logger import Component, PipelineStage

        logger = ErrorLogger()
        doc = logger._create_error_document(
            error=TimeoutError("x"),
            description="rag",
            component=Component.LLM,
            pipeline_stage=PipelineStage.GENERATION,
        )
        assert doc.component == "LLM"
        assert doc.pipeline_stage == "generation"
        assert doc.kind == "rag"

    def test_error_extra_merged_into_error_dict(self):
        logger = ErrorLogger()
        doc = logger._create_error_document(
            error=RuntimeError("boom"),
            description="retry",
            kind="policy",
            policy="retry",
            node="agent",
            error_extra={"attempt": 2},
        )
        assert doc.error["attempt"] == 2
        assert "args" not in doc.error

    def test_policy_standin_does_not_repeat_description(self):
        logger = ErrorLogger()
        doc = logger._create_error_document(
            error=PolicyEvent("Policy 'fallback' on node 'agent'"),
            description="Policy 'fallback' on node 'agent'",
            policy="fallback",
            source="runner",
        )
        assert doc.kind == "fallback"
        assert doc.policy is None
        assert doc.error is None
        assert doc.description == "Policy 'fallback' on node 'agent'"

    def test_attributes_node_not_copied_when_top_level_node_matches(self):
        logger = ErrorLogger()

        class NodeTimeoutError(Exception):
            def __init__(self):
                super().__init__("idle")
                self.node = "agent"

        doc = logger._create_error_document(
            error=NodeTimeoutError(),
            description="timed out",
            policy="timeout",
            node="agent",
        )
        assert doc.node == "agent"
        assert doc.error["type"] == "NodeTimeoutError"
        assert "attributes" not in doc.error

    def test_logger_name_dropped_when_it_equals_source(self):
        logger = ErrorLogger()
        doc = logger._create_error_document(
            error=RuntimeError("boom"),
            description="frontend crash",
            source="frontend",
            logger_name="frontend",
            kind="tool_error",
        )
        assert doc.source == "frontend"
        assert doc.logger_name is None
        assert doc.kind == "tool_error"

    @pytest.mark.asyncio
    async def test_persist_policy_event_writes_kind_policy(self):
        logger = ErrorLogger()
        logger.log_error = AsyncMock()
        with patch("src.utils.error_logger.get_error_logger", return_value=logger):
            await persist_policy_event(
                policy="retry",
                description="Policy 'retry' on node 'agent' (attempt 2)",
                node="agent",
                graph="react",
                attempt=2,
            )
        kwargs = logger.log_error.await_args.kwargs
        assert kwargs["kind"] == "retry"
        assert kwargs.get("policy") is None
        assert kwargs["node"] == "agent"
        assert kwargs["graph"] == "react"
        assert kwargs["error_extra"]["attempt"] == 2


class TestClassifyMcpToolResult:
    def test_is_error_flag(self):
        result = MagicMock()
        result.isError = True
        result.content = []
        result.structuredContent = None
        assert classify_mcp_tool_result(result)[0] == "is_error"

    def test_jsonrpc_error(self):
        signal, _desc = classify_mcp_tool_result(
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "nope"}}
        )
        assert signal == "is_error"

    def test_nested_result_is_error(self):
        signal, _desc = classify_mcp_tool_result(
            {"result": {"isError": True, "content": []}}
        )
        assert signal == "is_error"

    def test_structured_error_dict(self):
        result = MagicMock()
        result.isError = False
        result.is_error = False
        result.structuredContent = {"error": "qdrant down"}
        result.content = []
        signal, desc = classify_mcp_tool_result(result)
        assert signal == "structured"
        assert "qdrant down" in desc

    def test_json_text_structured_error(self):
        result = MagicMock()
        result.isError = False
        result.is_error = False
        result.structuredContent = None
        result.structured_content = None
        block = MagicMock()
        block.text = '{"error": "timeout"}'
        result.content = [block]
        signal, _desc = classify_mcp_tool_result(result)
        assert signal == "structured"

    def test_401_text_heuristic(self):
        result = MagicMock()
        result.isError = False
        result.is_error = False
        result.structuredContent = None
        result.structured_content = None
        block = MagicMock()
        block.text = "Error 401 authentication failed"
        result.content = [block]
        signal, desc = classify_mcp_tool_result(result)
        assert signal == "text"
        assert "401" in desc

    def test_ordinary_error_phrase_is_not_logged(self):
        result = MagicMock()
        result.isError = False
        result.is_error = False
        result.structuredContent = None
        result.structured_content = None
        block = MagicMock()
        block.text = "there was an error in the calculation"
        result.content = [block]
        assert classify_mcp_tool_result(result) is None

    def test_success_is_none(self):
        result = MagicMock()
        result.isError = False
        result.is_error = False
        result.structuredContent = None
        result.structured_content = None
        block = MagicMock()
        block.text = "ok"
        result.content = [block]
        assert classify_mcp_tool_result(result) is None


class TestErrorLoggingInterceptor:
    @pytest.mark.asyncio
    async def test_exception_logs_and_reraise(self):
        icept = ErrorLoggingInterceptor()
        request = MagicMock(name="search", server_name="wiley")
        request.name = "search"
        request.server_name = "wiley"
        error_logger = MagicMock()
        error_logger.log_error = AsyncMock()

        async def handler(_req):
            raise ConnectionError("refused")

        with patch(
            "src.services.agents.core.interceptors.get_error_logger",
            return_value=error_logger,
        ):
            with pytest.raises(ConnectionError):
                await icept(request, handler)

        kwargs = error_logger.log_error.await_args.kwargs
        assert kwargs["kind"] == "tool_error"
        assert kwargs["error_extra"]["signal"] == "exception"
        assert kwargs["node"] == "tools"

    @pytest.mark.asyncio
    async def test_is_error_logs_and_returns(self):
        icept = ErrorLoggingInterceptor()
        request = MagicMock()
        request.name = "search"
        request.server_name = "wiley"
        result = MagicMock()
        result.isError = True
        result.is_error = True
        result.structuredContent = None
        result.structured_content = None
        result.content = []
        error_logger = MagicMock()
        error_logger.log_error = AsyncMock()

        async def handler(_req):
            return result

        with patch(
            "src.services.agents.core.interceptors.get_error_logger",
            return_value=error_logger,
        ):
            out = await icept(request, handler)

        assert out is result
        kwargs = error_logger.log_error.await_args.kwargs
        assert kwargs["kind"] == "tool_error"
        assert kwargs["error_extra"]["signal"] == "is_error"
        assert isinstance(kwargs["error"], ToolReturnedError)

    @pytest.mark.asyncio
    async def test_401_text_logs_and_returns(self):
        icept = ErrorLoggingInterceptor()
        request = MagicMock()
        request.name = "search"
        request.server_name = "wiley"
        result = MagicMock()
        result.isError = False
        result.is_error = False
        result.structuredContent = None
        result.structured_content = None
        block = MagicMock()
        block.text = "Error 401 authentication failed"
        result.content = [block]
        error_logger = MagicMock()
        error_logger.log_error = AsyncMock()

        async def handler(_req):
            return result

        with patch(
            "src.services.agents.core.interceptors.get_error_logger",
            return_value=error_logger,
        ):
            out = await icept(request, handler)

        assert out is result
        kwargs = error_logger.log_error.await_args.kwargs
        assert kwargs["error_extra"]["signal"] == "text"


class TestTimedNodeRetryCallback:
    @pytest.mark.asyncio
    async def test_node_attempt_greater_than_one_emits_retry(self):
        base = graphs_base_module()
        graph = base.AgentGraph()
        called = []

        async def on_policy(**kwargs):
            called.append(kwargs)

        async def agent_fn(state):
            return {"ok": True}

        wrapper = graph.timed_node("agent", agent_fn, on_policy=on_policy)
        runtime = MagicMock()
        runtime.execution_info.node_attempt = 2
        await wrapper({}, runtime)
        assert called == [{"node": "agent", "policy": "retry", "attempt": 2}]

    @pytest.mark.asyncio
    async def test_first_attempt_does_not_emit_retry(self):
        base = graphs_base_module()
        graph = base.AgentGraph()
        called = []

        async def on_policy(**kwargs):
            called.append(kwargs)

        async def agent_fn(state):
            return state

        wrapper = graph.timed_node("agent", agent_fn, on_policy=on_policy)
        runtime = MagicMock()
        runtime.execution_info.node_attempt = 1
        await wrapper({}, runtime)
        assert called == []


class TestGraphErrorHandler:
    @pytest.mark.asyncio
    async def test_uncaught_routes_to_agent_fallback(self):
        base = graphs_base_module()
        graph = base.AgentGraph()
        called = []

        async def on_policy(**kwargs):
            called.append(kwargs)

        handler = graph.error_handler(
            on_policy=on_policy, fallback_llm=MagicMock(name="fallback")
        )
        error = MagicMock()
        error.node = "agent"
        error.error = TimeoutError("idle")
        result = await handler({"messages": []}, error)
        policies = [c.get("policy") for c in called]
        assert "error_handler" in policies or "timeout" in policies
        assert "fallback" in policies
        goto = getattr(result, "goto", None)
        if goto is None and isinstance(result, dict):
            goto = result.get("goto")
        assert goto == "agent_fallback"

    @pytest.mark.asyncio
    async def test_without_fallback_reraises(self):
        base = graphs_base_module()
        graph = base.AgentGraph()
        handler = graph.error_handler(on_policy=None, fallback_llm=None)
        error = MagicMock()
        error.node = "agent"
        error.error = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            await handler({"messages": []}, error)

    @pytest.mark.asyncio
    async def test_fallback_node_timeout_logs_timeout_and_does_not_loop(self):
        class NodeTimeoutError(Exception):
            pass

        base = graphs_base_module()
        graph = base.AgentGraph()
        called = []

        async def on_policy(**kwargs):
            called.append(kwargs)

        handler = graph.error_handler(
            on_policy=on_policy, fallback_llm=MagicMock(name="fallback")
        )
        error = MagicMock()
        error.node = "agent_fallback"
        error.error = NodeTimeoutError("idle")
        with pytest.raises(NodeTimeoutError):
            await handler({"messages": []}, error)
        assert [c.get("policy") for c in called] == ["timeout"]


class TestRetryOnTransient:
    def test_value_error_is_not_retried(self):
        base = graphs_base_module()
        assert base.retry_on_transient(ValueError("bad args")) is False

    def test_connection_error_is_retried(self):
        base = graphs_base_module()
        assert base.retry_on_transient(ConnectionError("refused")) is True


class TestRedaction:
    def test_jwt_and_bearer_stripped_from_error_document(self):
        logger = ErrorLogger()
        token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature"
        )
        doc = logger._create_error_document(
            error=RuntimeError(f"Bearer {token}"),
            description=f"Authorization: Bearer {token}",
            kind="tool_error",
            source="mcp",
            error_extra={"token": "secret-value"},
        )
        blob = str(doc.error) + doc.description
        assert "eyJ" not in blob
        assert "secret-value" not in blob
        assert "[REDACTED]" in blob

    def test_structured_error_false_is_not_a_failure(self):
        result = MagicMock()
        result.isError = False
        result.is_error = False
        result.structuredContent = {"error": False}
        result.structured_content = None
        result.content = []
        assert classify_mcp_tool_result(result) is None


class TestPolicyForUncaught:
    def test_node_timeout_is_timeout_policy(self):
        from src.services.agents.core.runner import _policy_for_uncaught, _node_from_exception

        class NodeTimeoutError(Exception):
            def __init__(self):
                super().__init__("idle")
                self.node = "agent_fallback"

        exc = NodeTimeoutError()
        assert _policy_for_uncaught(exc) == "timeout"
        assert _node_from_exception(exc) == "agent_fallback"

    def test_stdlib_timeout_is_run_timeout(self):
        from src.services.agents.core.runner import _policy_for_uncaught

        assert _policy_for_uncaught(TimeoutError("budget")) == "run_timeout"

    def test_circuit_open_does_not_persist_an_error_log(self):
        from src.services.agents.core.runner import _record_endpoint_failure

        persist = AsyncMock()
        manager = MagicMock()
        with patch(
            "src.services.agents.core.runner.persist_policy_event", persist
        ), patch(
            "src.services.agents.core.runner.get_shared_llm_manager",
            return_value=manager,
        ), patch(
            "src.services.agents.core.runner.is_endpoint_failure", return_value=True
        ):
            _record_endpoint_failure(
                {"answered": "main"}, TimeoutError("no first token")
            )

        manager.health.record_failure.assert_called_once()
        persist.assert_not_called()
