"""UI RAG params must win over whatever the model puts in the retrieve call.

The retrieval MCP tool advertises defaults in its docstring, so the model
happily emits its own ``k``, ``score_threshold`` and ``public_collections``.
Those are user settings, not model settings: the request values overwrite them.
"""

from typing import Any, Dict, List

import pytest

from src.constants import DEFAULT_K, DEFAULT_SCORE_THRESHOLD
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import (
    _is_rag_mcp_tool,
    _rag_mcp_tool_defaults,
    _tool_arg_names,
    _with_request_rag_defaults,
)

pytestmark = pytest.mark.no_db

_RETRIEVE_ARGS = (
    "query",
    "k",
    "score_threshold",
    "public_collections",
    "private_collections",
    "collection_ids",
    "filters",
    "temperature",
    "max_new_tokens",
    "embeddings_model",
    "llm_type",
)


class FakeTool:
    """Minimal stand-in for a langchain StructuredTool built from an MCP tool."""

    def __init__(self, name: str, arg_names=_RETRIEVE_ARGS):
        self.name = name
        self.args: Dict[str, Any] = {arg: {} for arg in arg_names}
        self.calls: List[Any] = []

    async def ainvoke(self, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(tool_input)
        return "ok"


class FakeLayeredTool(FakeTool):
    """Tool exposing every call layer ``_with_request_rag_defaults`` patches."""

    def invoke(self, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(tool_input)
        return "ok"

    async def coroutine(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return "ok"

    def func(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return "ok"


class RaisingArgsTool:
    """A malformed MCP tool whose ``args`` property blows up on access."""

    def __init__(self, name: str = "eve_retrieval_retrieve"):
        self.name = name
        self.calls: List[Any] = []

    @property
    def args(self) -> Dict[str, Any]:
        raise RuntimeError("schema build failed")

    async def ainvoke(self, tool_input: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(tool_input)
        return "ok"


def _request(**overrides: Any) -> GenerationRequest:
    defaults: Dict[str, Any] = {
        "query": "request query",
        "k": 3,
        "score_threshold": 0.42,
        "public_collections": ["ESA Earth Observation"],
    }
    defaults.update(overrides)
    return GenerationRequest(**defaults)


async def _call(tool: FakeTool, request: GenerationRequest, tool_input: Any) -> Any:
    bound = _with_request_rag_defaults(tool, request)
    await bound.ainvoke(tool_input)
    return tool.calls[-1]


class TestUiParamsWin:
    @pytest.mark.asyncio
    async def test_model_supplied_values_are_replaced(self):
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request()

        sent = await _call(
            tool,
            request,
            {
                "query": "model query",
                "k": 5,
                "score_threshold": 0.7,
                "public_collections": ["Wiley AI Gateway"],
            },
        )

        assert sent["k"] == 3
        assert sent["score_threshold"] == 0.42
        assert sent["public_collections"] == ["ESA Earth Observation"]

    @pytest.mark.asyncio
    async def test_missing_ui_params_are_injected(self):
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request()

        sent = await _call(tool, request, {"query": "model query"})

        assert sent["k"] == 3
        assert sent["score_threshold"] == 0.42
        assert sent["public_collections"] == ["ESA Earth Observation"]

    @pytest.mark.asyncio
    async def test_model_query_survives_and_is_only_filled_when_missing(self):
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request()

        sent = await _call(tool, request, {"query": "model query"})
        assert sent["query"] == "model query"

        sent = await _call(tool, request, {})
        assert sent["query"] == "request query"

    @pytest.mark.asyncio
    async def test_request_filters_overwrite_model_filters(self):
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request(filters={"year": {"gte": 2020}})

        sent = await _call(tool, request, {"query": "model query", "filters": {}})

        assert sent["filters"] == {"year": {"gte": 2020}}

    @pytest.mark.asyncio
    async def test_empty_public_collections_are_still_sent(self):
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request(public_collections=[])

        sent = await _call(
            tool, request, {"query": "q", "public_collections": ["Wiley AI Gateway"]}
        )

        assert sent["public_collections"] == []

    @pytest.mark.asyncio
    async def test_request_year_overwrites_the_model_year(self):
        tool = FakeTool(
            "eve_retrieval_retrieve",
            arg_names=_RETRIEVE_ARGS + ("year",),
        )
        request = _request(filters={"year": {"gte": 2020}}, year=[2020, 2024])

        sent = await _call(tool, request, {"query": "q", "year": [1999]})

        assert sent["year"] == [2020, 2024]

    @pytest.mark.asyncio
    async def test_model_year_is_cleared_when_the_request_has_none(self):
        tool = FakeTool(
            "eve_retrieval_retrieve",
            arg_names=_RETRIEVE_ARGS + ("year",),
        )
        request = _request()

        sent = await _call(tool, request, {"query": "q", "year": [1999]})

        assert sent["year"] is None

    @pytest.mark.asyncio
    async def test_year_range_aliases_are_never_injected(self):
        """``POST /retrieve`` recomputes the range, so only ``year`` is sent."""
        tool = FakeTool(
            "eve_retrieval_retrieve",
            arg_names=_RETRIEVE_ARGS + ("year", "start_year", "end_year"),
        )
        request = _request(year=[2020, 2024])

        sent = await _call(tool, request, {"query": "q"})

        assert "start_year" not in sent
        assert "end_year" not in sent

    @pytest.mark.asyncio
    async def test_model_filters_are_cleared_when_the_request_has_none(self):
        """No filter selected in the UI means no filter, not the model's guess."""
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request()

        sent = await _call(
            tool, request, {"query": "q", "filters": {"must": [{"key": "year"}]}}
        )

        assert sent["filters"] is None

    @pytest.mark.asyncio
    async def test_model_llm_type_is_cleared_when_the_request_has_none(self):
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request()

        sent = await _call(tool, request, {"query": "q", "llm_type": "satcom_large"})

        assert sent["llm_type"] is None

    @pytest.mark.asyncio
    async def test_request_llm_type_overwrites_the_model_llm_type(self):
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request(llm_type="main")

        sent = await _call(tool, request, {"query": "q", "llm_type": "satcom_large"})

        assert sent["llm_type"] == "main"


class TestSchemaIsRespected:
    @pytest.mark.asyncio
    async def test_arg_absent_from_schema_is_not_sent(self):
        arg_names = tuple(a for a in _RETRIEVE_ARGS if a != "private_collections")
        tool = FakeTool("eve_retrieval_retrieve", arg_names=arg_names)
        request = _request(private_collections=["68b0c0ffee0000000000dead"])

        sent = await _call(tool, request, {"query": "q"})

        assert "private_collections" not in sent

    @pytest.mark.asyncio
    async def test_private_collections_sent_when_schema_accepts_them(self):
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request(private_collections=["68b0c0ffee0000000000dead"])

        sent = await _call(tool, request, {"query": "q"})

        assert sent["private_collections"] == ["68b0c0ffee0000000000dead"]

    @pytest.mark.asyncio
    async def test_empty_private_collections_override_the_model(self):
        """No private collection selected means none searched, not the model's guess."""
        tool = FakeTool("eve_retrieval_retrieve")
        request = _request()

        sent = await _call(
            tool,
            request,
            {"query": "q", "private_collections": ["68b0c0ffee0000000000dead"]},
        )

        assert sent["private_collections"] == []

    @pytest.mark.asyncio
    async def test_non_introspectable_schema_injects_nothing(self):
        """An empty accepted set means unknown, not everything.

        Injecting on a guess would push every alias (k, top_k, top_n, limit,
        n_results) into a single call and blow up the tool invocation.
        """
        tool = FakeTool("eve_retrieval_retrieve", arg_names=())
        request = _request()

        sent = await _call(tool, request, {"query": "model query"})

        assert sent == {"query": "model query"}

    @pytest.mark.asyncio
    async def test_unrelated_tool_is_returned_untouched(self):
        tool = FakeTool("geocode_place", arg_names=("place",))
        request = _request()

        assert _with_request_rag_defaults(tool, request) is tool

        sent = await _call(tool, request, {"place": "Rome"})
        assert sent == {"place": "Rome"}


class TestDetection:
    def test_prefixed_literal_matches(self):
        assert _is_rag_mcp_tool(FakeTool("eve_retrieval_retrieve")) is True

    def test_other_server_prefix_with_rag_schema_matches(self):
        assert _is_rag_mcp_tool(FakeTool("eve_retrieval_v2_retrieve")) is True

    def test_retrieve_without_rag_schema_does_not_match(self):
        assert (
            _is_rag_mcp_tool(FakeTool("other_retrieve", arg_names=("query", "k")))
            is False
        )

    def test_generic_search_tool_does_not_match(self):
        assert (
            _is_rag_mcp_tool(
                FakeTool("esa_moocs_search_moocs", arg_names=("query", "top_k"))
            )
            is False
        )

    def test_unnamed_tool_does_not_match(self):
        assert _is_rag_mcp_tool(object()) is False


class TestDefaultsMap:
    def test_only_accepted_names_are_mapped(self):
        tool = FakeTool("eve_retrieval_retrieve", arg_names=("query", "top_k"))
        defaults = _rag_mcp_tool_defaults(tool, _request())

        assert set(defaults) == {"query", "top_k"}
        assert defaults["top_k"] == 3


class TestEveryCallLayerOverrides:
    """Whichever layer LangChain reaches, the UI params are the ones sent."""

    @pytest.mark.asyncio
    async def test_invoke_layer_applies_the_override(self):
        tool = FakeLayeredTool("eve_retrieval_retrieve")
        request = _request()

        bound = _with_request_rag_defaults(tool, request)
        bound.invoke({"query": "model query", "k": 5, "score_threshold": 0.9})

        sent = tool.calls[-1]
        assert sent["k"] == 3
        assert sent["score_threshold"] == 0.42
        assert sent["public_collections"] == ["ESA Earth Observation"]

    @pytest.mark.asyncio
    async def test_coroutine_layer_applies_the_override(self):
        tool = FakeLayeredTool("eve_retrieval_retrieve")
        request = _request()

        bound = _with_request_rag_defaults(tool, request)
        await bound.coroutine(query="model query", k=5, score_threshold=0.9)

        sent = tool.calls[-1]
        assert sent["k"] == 3
        assert sent["score_threshold"] == 0.42
        assert sent["public_collections"] == ["ESA Earth Observation"]

    @pytest.mark.asyncio
    async def test_func_layer_applies_the_override(self):
        tool = FakeLayeredTool("eve_retrieval_retrieve")
        request = _request()

        bound = _with_request_rag_defaults(tool, request)
        bound.func(query="model query", k=5, score_threshold=0.9)

        sent = tool.calls[-1]
        assert sent["k"] == 3
        assert sent["score_threshold"] == 0.42
        assert sent["public_collections"] == ["ESA Earth Observation"]


class TestFrontendDefaultsAreTheDefaults:
    """A request that carries no k/score_threshold still sends the UI defaults."""

    @pytest.mark.asyncio
    async def test_untouched_request_sends_ten_and_zero_point_six(self):
        tool = FakeTool("eve_retrieval_retrieve")
        request = GenerationRequest(
            query="request query", public_collections=["ESA Earth Observation"]
        )

        sent = await _call(tool, request, {"query": "model query"})

        assert sent["k"] == 10
        assert sent["score_threshold"] == 0.6

    def test_constants_match_the_frontend_default_set(self):
        assert DEFAULT_K == 10
        assert DEFAULT_SCORE_THRESHOLD == 0.6


class TestBrokenToolIsSurvived:
    """One malformed tool must not take the whole tool list down with it."""

    def test_raising_args_property_yields_no_argument_names(self):
        assert _tool_arg_names(RaisingArgsTool()) == set()

    def test_tool_with_raising_args_is_returned_untouched(self):
        tool = RaisingArgsTool()

        assert _with_request_rag_defaults(tool, _request()) is tool

    @pytest.mark.asyncio
    async def test_tool_with_raising_args_still_calls_through(self):
        tool = RaisingArgsTool()

        sent = await _call(tool, _request(), {"query": "model query"})

        assert sent == {"query": "model query"}
