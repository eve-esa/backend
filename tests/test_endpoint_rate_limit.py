"""A 429 from a model endpoint parks it for the Retry-After it asked for.

Blablador (JSC) doubles a block on every request it receives during the block,
from 60 s up to an hour, so a retry or a next turn sent into the window makes
the outage longer. These tests pin what the backend does instead: one request,
then the fallback model for the rest of the turn and for every turn until the
window ends, then one probe.
"""

import asyncio
import email.utils
import json
import logging

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APIStatusError

from src.core.llm_health import EndpointHealth, EndpointRateLimited, reset_shared_stores
from src.core.llm_manager import BreakerAwareChatOpenAI, LLMManager, LLMType
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core import runner as runner_module
from src.services.agents.graphs_bundle import graphs_prefix
from tests.test_llm_health import _Clock, _FakeRedis

pytestmark = pytest.mark.no_db

JSC = LLMType.Eve_Jsc.value
FALLBACK = LLMType.Fallback.value
SECRET_KEY = "jsc-secret-key-do-not-log"
USER_TEXT = "my private question about the flood map"
KEY = "eve:breaker:eve_jsc"


@pytest.fixture(autouse=True)
def _fresh_stores():
    reset_shared_stores()
    yield
    reset_shared_stores()


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://jsc.example/v1/chat/completions")


def _rate_limited(headers=None) -> APIStatusError:
    # Blablador echoes context in its error body: none of it may reach a log.
    body = {"error": {"message": f"Too many requests for {SECRET_KEY}: {USER_TEXT}"}}
    return APIStatusError(
        f"Error code: 429 - {body}",
        response=httpx.Response(429, headers=headers or {}, json=body, request=_request()),
        body=body,
    )


def _status_error(status: int) -> APIStatusError:
    return APIStatusError(
        "upstream said no",
        response=httpx.Response(status, request=_request()),
        body=None,
    )


def _health(clock: _Clock, store=None) -> EndpointHealth:
    return EndpointHealth(
        cooldown_s=120, clock=clock, wall_clock=clock, redis_client=store, probe_s=15
    )


def _clock(start: float = 1000.0) -> _Clock:
    clock = _Clock()
    clock.now = start
    return clock


# ─── window length ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("headers", "window"),
    [
        ({"retry-after": "30"}, 30),
        ({}, 60),
        ({"retry-after": "soon"}, 60),
        ({"retry-after": "86400"}, 3600),
    ],
    ids=["seconds", "missing", "unparsable", "capped"],
)
def test_a_429_opens_the_circuit_for_its_retry_after(headers, window):
    clock = _clock()
    health = _health(clock)

    health.record_failure(JSC, _rate_limited(headers))

    clock.now = 1000.0 + window - 0.5
    assert health.is_open(JSC) is True
    assert health.is_rate_limited(JSC) is True
    clock.now = 1000.0 + window
    assert health.is_open(JSC) is False
    assert health.is_rate_limited(JSC) is False


def test_an_http_date_retry_after_counts_from_the_wall_clock():
    clock = _clock(1_800_000_000.0)
    health = _health(clock)
    date = email.utils.formatdate(1_800_000_000.0 + 90, usegmt=True)

    health.record_failure(JSC, _rate_limited({"retry-after": date}))

    clock.now += 89.5
    assert health.is_open(JSC) is True
    clock.now += 0.5
    assert health.is_open(JSC) is False


def test_the_shared_circuit_carries_the_window_to_every_worker():
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)

    worker_a.record_failure(JSC, _rate_limited({"retry-after": "30"}))

    assert worker_b.is_rate_limited(JSC) is True
    assert worker_b.rate_limit_remaining_s(JSC) == pytest.approx(30.0)
    clock.now = 1029.0
    assert worker_b.is_rate_limited(JSC) is True
    # Window over: one worker probes, the others stay off the endpoint.
    clock.now = 1030.0
    verdicts = sorted([worker_a.is_rate_limited(JSC), worker_b.is_rate_limited(JSC)])
    assert verdicts == [False, True]


def test_a_timeout_never_shortens_a_window_the_endpoint_asked_for():
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "600"}))

    clock.now = 1010.0
    worker_b.record_failure(JSC, TimeoutError("first token budget"))

    clock.now = 1300.0
    assert worker_a.is_rate_limited(JSC) is True
    assert worker_b.is_rate_limited(JSC) is True


def test_non_429_failures_keep_the_fixed_cooldown():
    clock = _clock()
    health = _health(clock)

    for exc in (_status_error(500), TimeoutError("first token budget")):
        health.record_failure(JSC, exc)
        assert health.is_open(JSC) is True
        assert health.is_rate_limited(JSC) is False
        clock.now += 119.0
        assert health.is_open(JSC) is True
        clock.now += 1.0
        assert health.is_open(JSC) is False


def test_the_log_line_names_the_window_and_nothing_else(caplog):
    health = _health(_clock())

    with caplog.at_level(logging.WARNING, logger="src.core.llm_health"):
        health.record_failure(JSC, _rate_limited({"retry-after": "30"}))

    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(lines) == 1
    assert lines[0].startswith("Endpoint eve_jsc circuit opened:")
    assert "429" in lines[0] and "30s" in lines[0]
    assert SECRET_KEY not in lines[0] and USER_TEXT not in lines[0]


# ─── chain resolution ─────────────────────────────────────────────────────────


def _manager(monkeypatch, health: EndpointHealth) -> LLMManager:
    manager = LLMManager()
    manager._health = health
    monkeypatch.setattr(manager, "_is_configured", lambda name: name in {JSC, "main", FALLBACK})
    return manager


def test_an_explicit_request_skips_a_rate_limited_endpoint(monkeypatch):
    health = _health(_clock())
    manager = _manager(monkeypatch, health)
    health.record_failure(JSC, _rate_limited({"retry-after": "30"}))

    assert manager.resolve_chain(JSC) == [FALLBACK]
    assert JSC not in manager.resolve_chain(None)
    assert manager.open_circuits(JSC, [FALLBACK]) == [JSC]


def test_an_explicit_request_keeps_an_endpoint_after_a_timeout(monkeypatch):
    health = _health(_clock())
    manager = _manager(monkeypatch, health)
    health.record_failure(JSC, TimeoutError("first token budget"))

    assert manager.resolve_chain(JSC) == [JSC, FALLBACK]


def test_get_client_for_model_substitutes_inside_the_window(monkeypatch):
    health = _health(_clock())
    manager = _manager(monkeypatch, health)
    fallback_client, jsc_client = object(), object()
    monkeypatch.setattr(manager, "_get_fallback_llm", lambda: fallback_client)
    monkeypatch.setattr(manager, "_get_eve_jsc_llm", lambda: jsc_client)
    health.record_failure(JSC, _rate_limited({"retry-after": "30"}))

    assert manager.get_client_for_model(JSC) is fallback_client
    assert manager.get_selected_llm_type() == FALLBACK


async def test_a_new_turn_inside_the_window_resolves_to_the_fallback(monkeypatch):
    health = _health(_clock())
    manager = _manager(monkeypatch, health)
    monkeypatch.setattr(manager, "_get_fallback_llm", lambda: "fallback-client")
    monkeypatch.setattr(
        manager, "_get_eve_jsc_llm", lambda: pytest.fail("JSC client built inside the window")
    )
    monkeypatch.setattr(runner_module, "get_shared_llm_manager", lambda: manager)
    health.record_failure(JSC, _rate_limited({"retry-after": "30"}))

    llm, meta = await runner_module._resolve_agentic_llm_client(
        GenerationRequest(query="q", llm_type=JSC), user_id="u1"
    )

    assert llm == "fallback-client"
    endpoint = meta["endpoint"]
    assert endpoint["requested"] == JSC
    assert endpoint["answered"] == FALLBACK
    assert endpoint["substituted"] is True
    assert endpoint["circuit_open"] == [JSC]


# ─── the client and the agent graph ───────────────────────────────────────────


class _Upstream:
    """An OpenAI-compatible endpoint behind httpx.MockTransport, counting calls."""

    def __init__(self, answer):
        self.calls = 0
        self._answer = answer

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self._answer(json.loads(request.content), self.calls)


def _completion(message: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", **message},
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _agent_turn(body: dict, _calls: int) -> httpx.Response:
    """Call the tool once, then answer.

    The react graph rewrites tool results as text for text-format models, so
    the tool's output is looked up anywhere in the request.
    """
    if "found it" in json.dumps(body["messages"]):
        return _completion({"content": "The answer."})
    return _completion(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query": "flood"}'},
                }
            ],
        }
    )


def _jsc_429(headers=None):
    def answer(body: dict, _calls: int) -> httpx.Response:
        return httpx.Response(
            429,
            headers=headers or {"retry-after": "30"},
            json={"error": {"message": f"Too many requests for {SECRET_KEY}"}},
        )

    return answer


def _jsc_status(status: int):
    def answer(body: dict, _calls: int) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "upstream exploded"}})

    return answer


def _client(cls, upstream: _Upstream, base_url: str):
    return cls(
        api_key="test-key",
        base_url=base_url,
        model="m",
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
    )


def _jsc_client(upstream: _Upstream, health: EndpointHealth) -> BreakerAwareChatOpenAI:
    client = _client(BreakerAwareChatOpenAI, upstream, "https://jsc.example/v1")
    client.guard(health, JSC)
    return client


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return "found it"


async def _run_react_turn(llm, fallback_llm) -> str:
    import importlib

    react = importlib.import_module(f"{graphs_prefix()}.react.graph")
    graph = react.ReactAgent().compile(
        llm=llm,
        tools=[lookup],
        checkpointer=None,
        fallback_llm=fallback_llm,
        llm_idle_timeout=15,
        llm_run_timeout=60,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(USER_TEXT)]})
    return result["messages"][-1].content


async def test_an_agentic_turn_sends_one_jsc_request_then_stays_on_the_fallback():
    health = _health(_clock())
    jsc = _Upstream(_jsc_429())
    mistral = _Upstream(_agent_turn)

    answer = await _run_react_turn(
        _jsc_client(jsc, health), _client(ChatOpenAI, mistral, "https://fallback.example/v1")
    )

    assert answer == "The answer."
    # No node retry into the block, and the second agent step skips JSC.
    assert jsc.calls == 1
    assert mistral.calls == 2
    assert health.is_rate_limited(JSC) is True


async def test_turns_inside_the_window_send_nothing_and_the_probe_sends_one(monkeypatch):
    clock = _clock()
    health = _health(clock)
    manager = _manager(monkeypatch, health)
    jsc = _Upstream(_jsc_429())
    jsc_client = _jsc_client(jsc, health)
    mistral = _Upstream(_agent_turn)
    fallback_client = _client(ChatOpenAI, mistral, "https://fallback.example/v1")
    monkeypatch.setattr(manager, "_get_eve_jsc_llm", lambda: jsc_client)
    monkeypatch.setattr(manager, "_get_fallback_llm", lambda: fallback_client)
    monkeypatch.setattr(runner_module, "get_shared_llm_manager", lambda: manager)
    request = GenerationRequest(query="q", llm_type=JSC)

    async def turn():
        llm, meta = await runner_module._resolve_agentic_llm_client(request, user_id="u1")
        return await _run_react_turn(llm, fallback_client), meta["endpoint"]

    await turn()
    assert jsc.calls == 1

    clock.now = 1015.0
    answer, endpoint = await turn()
    assert answer == "The answer."
    assert jsc.calls == 1
    assert endpoint["substituted"] is True and endpoint["answered"] == FALLBACK

    # Window over: the next turn goes back to JSC (its first request is the
    # probe, then the tool step), and the answer closes the circuit.
    clock.now = 1030.0
    jsc._answer = _agent_turn
    answer, endpoint = await turn()
    assert answer == "The answer."
    assert jsc.calls == 3
    assert endpoint["answered"] == JSC
    assert health.is_open(JSC) is False


@pytest.mark.parametrize(("status", "jsc_calls"), [(503, 4), (500, 2)])
async def test_a_jsc_5xx_keeps_todays_retry_and_cooldown(status, jsc_calls):
    health = _health(_clock())
    jsc = _Upstream(_jsc_status(status))
    mistral = _Upstream(_agent_turn)

    answer = await _run_react_turn(
        _jsc_client(jsc, health), _client(ChatOpenAI, mistral, "https://fallback.example/v1")
    )

    assert answer == "The answer."
    # As before: a 503 is retried in place at both steps, a 500 is not, and an
    # in-graph fallback leaves the circuit alone.
    assert jsc.calls == jsc_calls
    assert health.is_open(JSC) is False


async def test_the_client_turns_a_429_into_a_recorded_rate_limit():
    health = _health(_clock())
    jsc = _Upstream(_jsc_429({"retry-after": "45"}))
    client = _jsc_client(jsc, health)

    with pytest.raises(EndpointRateLimited) as raised:
        await client.ainvoke("hi")
    assert raised.value.retry_after_s == pytest.approx(45.0)
    assert raised.value.status_code == 429
    assert not hasattr(raised.value, "response")
    assert health.rate_limit_remaining_s(JSC) == pytest.approx(45.0)

    # Inside the window the client refuses without sending a request.
    with pytest.raises(EndpointRateLimited):
        await client.ainvoke("hi again")
    assert jsc.calls == 1


async def test_recording_the_translated_error_again_changes_nothing(caplog):
    clock = _clock()
    health = _health(clock)
    jsc = _Upstream(_jsc_429({"retry-after": "45"}))

    with caplog.at_level(logging.WARNING, logger="src.core.llm_health"):
        with pytest.raises(EndpointRateLimited) as raised:
            await _jsc_client(jsc, health).ainvoke("hi")
        # The classic walk and the agentic failure path record what they catch.
        health.record_failure(JSC, raised.value)

    opened = [r for r in caplog.records if "circuit opened" in r.getMessage()]
    assert len(opened) == 1
    assert health.rate_limit_remaining_s(JSC) == pytest.approx(45.0)


async def test_a_stop_during_the_call_is_not_a_rate_limit():
    health = _health(_clock())
    arrived = asyncio.Event()

    async def hang(request):
        arrived.set()
        await asyncio.Event().wait()

    client = BreakerAwareChatOpenAI(
        api_key="k",
        base_url="https://jsc.example/v1",
        model="m",
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(hang)),
    )
    client.guard(health, JSC)

    task = asyncio.create_task(client.ainvoke("hi"))
    await asyncio.wait_for(arrived.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert health.is_open(JSC) is False
