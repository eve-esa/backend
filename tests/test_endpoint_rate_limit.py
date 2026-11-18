"""A 429 from a model endpoint parks it for its Retry-After, backed off.

Blablador (JSC) doubles a block on every request it receives during the block,
from 60 s up to an hour, so a retry or a next turn sent into the window makes
the outage longer. These tests pin what the backend does instead: one request,
then the fallback model for the rest of the turn and for every turn until the
window ends, then one probe.
"""

import asyncio
import contextlib
import email.utils
import json
import logging
import os
import traceback
import uuid
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APIStatusError

from src.core.llm_health import (
    PROBE_LEASE_S,
    EndpointHealth,
    EndpointRateLimited,
    is_endpoint_failure,
    parse_retry_after,
    rate_limit_backoff_s,
    reset_shared_stores,
)
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


def _remaining(health: EndpointHealth, llm_type: str = JSC) -> float:
    """Seconds left in the window as *health* reads it, 0 outside one."""
    view = health._window_view(llm_type)
    return max(view.until - health._wall(), 0.0) if view is not None else 0.0


# ─── Retry-After ──────────────────────────────────────────────────────────────

NOW = 1_800_000_000.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("30", 30.0),
        (" 30 ", 30.0),
        ("0", 0.0),
        ("1.5", 1.5),
        ("99999999999999999999", 1e20),
        ("\u00b2", None),
        ("\u0663\u0660", None),
        ("-5", None),
        ("1e3", None),
        ("inf", None),
        ("nan", None),
        ("soon", None),
        ("", None),
        (None, None),
        (email.utils.formatdate(NOW + 90, usegmt=True), 90.0),
        (email.utils.formatdate(NOW - 90, usegmt=True), None),
        ("Fri, 31 Dec 9999 23:59:59 GMT", pytest.approx(253402300799 - NOW)),
    ],
    ids=[
        "seconds", "spaces", "zero", "decimal", "huge", "superscript", "arabic-indic",
        "negative", "exponent", "inf", "nan", "garbage", "empty", "none",
        "future-date", "past-date", "far-date",
    ],
)
def test_parse_retry_after(raw, expected):
    assert parse_retry_after(raw, NOW) == expected


def test_the_reranker_reads_retry_after_with_the_same_parser():
    from src.utils import jsc_reranker

    assert jsc_reranker.parse_retry_after is parse_retry_after
    assert jsc_reranker._retry_after_seconds("\u00b2") == jsc_reranker.RETRY_AFTER_DEFAULT_S


# ─── window length ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("headers", "window"),
    [
        ({"retry-after": "90"}, 90),
        ({"retry-after": "30"}, 60),
        ({}, 60),
        ({"retry-after": "soon"}, 60),
        ({"retry-after": "-5"}, 60),
        ({"retry-after": "1.5"}, 60),
        ({"retry-after": "86400"}, 3600),
        ({"retry-after": "99999999999999999999"}, 3600),
        ({"retry-after": email.utils.formatdate(900.0, usegmt=True)}, 60),
    ],
    ids=["seconds", "below-the-base", "missing", "unparsable", "negative", "decimal",
         "capped", "huge", "past-date"],
)
def test_a_first_429_opens_the_circuit_for_its_retry_after_or_the_base(headers, window):
    clock = _clock()
    health = _health(clock)

    assert health.record_failure(JSC, _rate_limited(headers)) == pytest.approx(window)

    clock.now = 1000.0 + window - 0.5
    assert health.is_open(JSC) is True
    assert health.is_rate_limited(JSC) is True
    clock.now = 1000.0 + window
    assert health.is_open(JSC) is False
    assert health.is_rate_limited(JSC) is False


def test_a_unicode_digit_retry_after_falls_back_to_the_default():
    # httpx refuses non-ASCII header values, so the header arrives as text
    # some other way: "\u00b2".isdigit() is True and float() of it raises.
    exc = SimpleNamespace(
        status_code=429, response=SimpleNamespace(headers={"retry-after": "\u00b2"})
    )
    health = _health(_clock())

    assert health.record_failure(JSC, exc) == pytest.approx(60)


def test_an_http_date_retry_after_counts_from_the_wall_clock():
    clock = _clock(1_800_000_000.0)
    health = _health(clock)
    date = email.utils.formatdate(1_800_000_000.0 + 90, usegmt=True)

    health.record_failure(JSC, _rate_limited({"retry-after": date}))

    clock.now += 89.5
    assert health.is_open(JSC) is True
    clock.now += 0.5
    assert health.is_open(JSC) is False


def test_the_shared_window_reaches_every_worker_and_admits_one_probe():
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)

    worker_a.record_failure(JSC, _rate_limited({"retry-after": "90"}))

    assert worker_b.is_rate_limited(JSC) is True
    assert _remaining(worker_b) == pytest.approx(90.0)
    with pytest.raises(EndpointRateLimited):
        worker_b.admit(JSC)
    clock.now = 1089.0
    assert worker_b.is_rate_limited(JSC) is True
    # Window over: nobody probes yet, so a new turn may try; reading never
    # takes the lease.
    clock.now = 1090.0
    assert [worker_a.is_rate_limited(JSC), worker_b.is_rate_limited(JSC)] == [False, False]
    # Exactly one request in the fleet is the probe.
    assert worker_a.admit(JSC).probe is True
    with pytest.raises(EndpointRateLimited):
        worker_b.admit(JSC)
    with pytest.raises(EndpointRateLimited):
        worker_a.admit(JSC)
    clock.now = 1091.5
    assert worker_b.is_rate_limited(JSC) is True


@pytest.mark.parametrize("retry_after", ["60", "600"])
@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
@pytest.mark.parametrize(
    "failure",
    [TimeoutError("first token budget"), _status_error(503)],
    ids=["timeout", "5xx"],
)
def test_a_timeout_or_5xx_never_shortens_a_window(retry_after, shared, failure):
    clock = _clock()
    store = _FakeRedis(clock) if shared else None
    worker_a = _health(clock, store)
    worker_b = _health(clock, store) if shared else worker_a
    worker_a.record_failure(JSC, _rate_limited({"retry-after": retry_after}))

    # An in-flight request that started before the 429 fails afterwards.
    clock.now = 1010.0
    worker_b.record_failure(JSC, failure)

    clock.now = 1000.0 + int(retry_after) - 1
    for worker in (worker_a, worker_b):
        assert worker.is_rate_limited(JSC) is True
        with pytest.raises(EndpointRateLimited):
            worker.admit(JSC)


@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
def test_a_shorter_429_never_shortens_a_window(shared):
    clock = _clock()
    store = _FakeRedis(clock) if shared else None
    worker_a = _health(clock, store)
    worker_b = _health(clock, store) if shared else worker_a
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "600"}))

    clock.now = 1010.0
    assert worker_b.record_failure(JSC, _rate_limited({"retry-after": "30"})) == pytest.approx(590)

    clock.now = 1100.0
    assert worker_a.is_rate_limited(JSC) is True
    assert worker_b.is_rate_limited(JSC) is True


def test_a_window_recorded_while_the_store_was_down_wins_on_reconnect():
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)
    store.down = True
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "600"}))
    store.down = False
    # Meanwhile another worker writes a shorter window.
    worker_b.record_failure(JSC, _rate_limited({"retry-after": "30"}))

    clock.now = 1040.0  # past the store retry window: worker A writes back first
    assert worker_a.is_rate_limited(JSC) is True
    clock.now = 1100.0
    assert worker_b.is_rate_limited(JSC) is True
    with pytest.raises(EndpointRateLimited):
        worker_b.admit(JSC)


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


@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
def test_a_success_reported_for_the_turn_never_closes_the_window(shared):
    clock = _clock()
    store = _FakeRedis(clock) if shared else None
    health = _health(clock, store)
    health.record_failure(JSC, _rate_limited({"retry-after": "30"}))

    # The classic walk reports the candidate it asked for, whatever client
    # answered; only the endpoint's own probe closes a window.
    clock.now = 1005.0
    health.record_success(JSC, started_at=1004.0)

    assert health.is_rate_limited(JSC) is True


@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
def test_only_the_probe_closes_the_window_and_a_newer_429_survives_it(shared):
    clock = _clock()
    store = _FakeRedis(clock) if shared else None
    health = _health(clock, store)
    health.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    clock.now = 1060.0
    permit = health.admit(JSC)
    assert permit.probe is True

    # A request that started before the window opened does not close it.
    assert health.close_rate_limit_window(JSC, started_at=999.0) is False
    assert health.close_rate_limit_window(JSC, started_at=permit.started_at) is True
    assert health.is_rate_limited(JSC) is False
    assert health.admit(JSC).probe is False


@pytest.mark.parametrize("probe_s", [5, 300], ids=["short-budget", "long-budget"])
@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
def test_the_probe_lease_has_one_fixed_length(shared, probe_s):
    clock = _clock()
    store = _FakeRedis(clock) if shared else None

    def worker():
        return EndpointHealth(
            cooldown_s=120, clock=clock, wall_clock=clock, redis_client=store, probe_s=probe_s
        )

    worker_a = worker()
    worker_b = worker() if shared else worker_a
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    clock.now = 1060.0
    assert worker_a.admit(JSC).probe is True

    # A probe that never reports (a Stop, a crash) holds the endpoint for the
    # lease, whatever the endpoint's first-token budget.
    clock.now = 1060.0 + PROBE_LEASE_S - 0.5
    with pytest.raises(EndpointRateLimited):
        worker_b.admit(JSC)
    clock.now = 1060.0 + PROBE_LEASE_S
    assert worker_b.admit(JSC).probe is True


def test_the_log_line_names_the_window_and_nothing_else(caplog):
    health = _health(_clock())

    with caplog.at_level(logging.WARNING, logger="src.core.llm_health"):
        health.record_failure(JSC, _rate_limited({"retry-after": "90"}))

    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(lines) == 1
    assert lines[0].startswith("Endpoint eve_jsc circuit opened:")
    assert "429" in lines[0] and "90s" in lines[0]
    assert SECRET_KEY not in lines[0] and USER_TEXT not in lines[0]


# The window commands against a real Redis (CI and the gate run one). The
# breaker passes the clock in, so the test clock drives them; only key expiry
# is real.


def _real_redis():
    url = os.getenv("REDIS_URL", "")
    if not url:
        pytest.skip("REDIS_URL not set")
    import redis

    client = redis.Redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=0.5)
    try:
        client.ping()
    except Exception:
        pytest.skip("no Redis answering at REDIS_URL")
    return client


def test_the_window_commands_on_a_real_redis():
    client = _real_redis()
    prefix = f"eve:test:{uuid.uuid4().hex}:"
    clock = _clock(1_800_000_000.0)

    def worker():
        return EndpointHealth(
            cooldown_s=120, clock=clock, wall_clock=clock, redis_client=client,
            probe_s=15, key_prefix=prefix,
        )

    worker_a, worker_b = worker(), worker()
    try:
        assert worker_a.admit(JSC).probe is False
        worker_a.record_failure(JSC, _rate_limited({"retry-after": "600"}))
        worker_b.record_failure(JSC, _rate_limited({"retry-after": "30"}))
        worker_b.record_failure(JSC, TimeoutError("in flight"))
        clock.now += 300
        with pytest.raises(EndpointRateLimited) as raised:
            worker_b.admit(JSC)
        assert raised.value.retry_after_s == pytest.approx(300, abs=1)

        clock.now += 300
        assert worker_b.is_rate_limited(JSC) is False
        permit = worker_a.admit(JSC)
        assert permit.probe is True
        with pytest.raises(EndpointRateLimited):
            worker_b.admit(JSC)
        assert worker_b.close_rate_limit_window(JSC, started_at=permit.started_at - 900) is False
        assert worker_a.close_rate_limit_window(JSC, started_at=permit.started_at) is True
        assert worker_b.admit(JSC).probe is False
    finally:
        for key in client.scan_iter(f"{prefix}*"):
            client.delete(key)


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


def test_reading_the_chain_after_a_window_never_takes_the_probe_lease(monkeypatch):
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)
    manager = _manager(monkeypatch, worker_a)
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "60"}))

    clock.now = 1060.0
    # Metadata and chain reads on a turn that may never call JSC.
    manager.open_circuits(None, [FALLBACK])
    manager.open_circuits(JSC, [FALLBACK])
    manager.resolve_chain(FALLBACK)

    assert worker_b.admit(JSC).probe is True


def test_the_hallucination_detector_shares_the_breaker(monkeypatch):
    from src.services import generate_answer
    from src.services.hallucination_detector import HallucinationDetector

    shared = LLMManager()
    monkeypatch.setattr(generate_answer, "get_shared_llm_manager", lambda: shared)

    assert HallucinationDetector().llm_manager.health is shared.health


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
    clock.now = 1060.0
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
    jsc = _Upstream(_jsc_429({"retry-after": "75"}))
    client = _jsc_client(jsc, health)

    with pytest.raises(EndpointRateLimited) as raised:
        await client.ainvoke("hi")
    assert raised.value.retry_after_s == pytest.approx(75.0)
    assert raised.value.status_code == 429
    assert not hasattr(raised.value, "response")
    assert _remaining(health) == pytest.approx(75.0)

    # Inside the window the client refuses without sending a request.
    with pytest.raises(EndpointRateLimited):
        await client.ainvoke("hi again")
    assert jsc.calls == 1


async def test_recording_the_translated_error_again_changes_nothing(caplog):
    clock = _clock()
    health = _health(clock)
    jsc = _Upstream(_jsc_429({"retry-after": "75"}))

    with caplog.at_level(logging.WARNING, logger="src.core.llm_health"):
        with pytest.raises(EndpointRateLimited) as raised:
            await _jsc_client(jsc, health).ainvoke("hi")
        # The classic walk and the agentic failure path record what they catch.
        health.record_failure(JSC, raised.value)

    opened = [r for r in caplog.records if "circuit opened" in r.getMessage()]
    assert len(opened) == 1
    assert _remaining(health) == pytest.approx(75.0)


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


async def test_the_translated_429_keeps_no_upstream_error_attached():
    health = _health(_clock())
    jsc = _Upstream(_jsc_429())

    with pytest.raises(EndpointRateLimited) as raised:
        await _jsc_client(jsc, health).ainvoke("hi")

    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert SECRET_KEY not in "".join(traceback.format_exception(raised.value))


def _jsc_plain_text_200(body: dict, _calls: int) -> httpx.Response:
    # Blablador has answered throttling with an error text inside a 200.
    return httpx.Response(
        200, content=b"Rate limit exceeded", headers={"content-type": "text/plain"}
    )


async def test_an_error_text_inside_a_200_fails_the_call_but_is_not_a_429():
    health = _health(_clock())
    jsc = _Upstream(_jsc_plain_text_200)

    with pytest.raises(Exception) as raised:
        await _jsc_client(jsc, health).ainvoke("hi")

    assert not isinstance(raised.value, EndpointRateLimited)
    assert getattr(raised.value, "status_code", None) != 429
    # As on main: not an endpoint failure, so no cooldown, and no window.
    assert is_endpoint_failure(raised.value) is False
    health.record_failure(JSC, raised.value)
    assert health.is_rate_limited(JSC) is False


async def test_a_turn_whose_jsc_answers_an_error_text_in_a_200_ends_on_the_fallback():
    health = _health(_clock())
    jsc = _Upstream(_jsc_plain_text_200)
    mistral = _Upstream(_agent_turn)

    answer = await _run_react_turn(
        _jsc_client(jsc, health), _client(ChatOpenAI, mistral, "https://fallback.example/v1")
    )

    assert answer == "The answer."
    assert mistral.calls == 2
    assert health.is_rate_limited(JSC) is False


# ─── the guard across workers ─────────────────────────────────────────────────


def _answer(_body: dict, _calls: int) -> httpx.Response:
    return _completion({"content": "ok"})


@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
async def test_while_the_probe_is_out_every_other_client_refuses(shared):
    clock = _clock()
    store = _FakeRedis(clock) if shared else None
    worker_a = _health(clock, store)
    worker_b = _health(clock, store) if shared else worker_a
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    jsc_b = _Upstream(_answer)
    client_b = _jsc_client(jsc_b, worker_b)

    # Window over, and worker A's request is the probe, still running.
    clock.now = 1060.0
    permit = worker_a.admit(JSC)
    assert permit.probe is True

    # A running turn on worker B holds its client: it must not send.
    with pytest.raises(EndpointRateLimited):
        await client_b.ainvoke("hi")
    assert jsc_b.calls == 0
    assert worker_b.is_rate_limited(JSC) is True

    # The probe answered: everyone may send again.
    worker_a.close_rate_limit_window(JSC, permit.started_at)
    clock.now = 1062.0
    assert (await client_b.ainvoke("hi")).content == "ok"
    assert jsc_b.calls == 1


async def test_after_the_window_one_client_probes_and_its_answer_closes_it():
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    jsc = _Upstream(_answer)

    clock.now = 1060.0
    assert (await _jsc_client(jsc, worker_b).ainvoke("hi")).content == "ok"

    assert jsc.calls == 1
    assert worker_a.admit(JSC).probe is False
    assert store.peek(KEY + ":rate-limit") is None


async def test_a_429_another_worker_just_recorded_holds_at_once():
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)
    jsc_b = _Upstream(_answer)
    client_b = _jsc_client(jsc_b, worker_b)
    # Worker B has just read the endpoint as healthy, and caches that.
    assert worker_b.is_rate_limited(JSC) is False

    worker_a.record_failure(JSC, _rate_limited({"retry-after": "60"}))

    with pytest.raises(EndpointRateLimited):
        await client_b.ainvoke("hi")
    assert jsc_b.calls == 0


async def test_with_the_store_down_the_guard_falls_back_to_the_local_window():
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)
    store.down = True
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    jsc_a, jsc_b = _Upstream(_answer), _Upstream(_answer)

    # Worker B never saw the 429: as before, it sends.
    assert (await _jsc_client(jsc_b, worker_b).ainvoke("hi")).content == "ok"
    with pytest.raises(EndpointRateLimited):
        await _jsc_client(jsc_a, worker_a).ainvoke("hi")
    assert (jsc_a.calls, jsc_b.calls) == (0, 1)


async def test_a_streaming_client_takes_the_lease_once_per_call(monkeypatch):
    clock = _clock()
    health = _health(clock)
    health.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    clock.now = 1060.0
    admitted = []
    real_admit = health.admit
    monkeypatch.setattr(health, "admit", lambda t: admitted.append(t) or real_admit(t))

    def stream(_body, _calls):
        body = (
            b'data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m",'
            b'"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},'
            b'"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    jsc = _Upstream(stream)
    client = _client(BreakerAwareChatOpenAI, jsc, "https://jsc.example/v1")
    client.streaming = True
    client.guard(health, JSC)

    assert (await client.ainvoke("hi")).content == "ok"
    assert admitted == [JSC]
    assert jsc.calls == 1
    assert health.is_rate_limited(JSC) is False


# ─── full turns ───────────────────────────────────────────────────────────────


async def _run_deep_turn(llm, fallback_llm) -> str:
    from agents.graphs.deep.graph import DeepAgent

    graph = DeepAgent().compile(
        llm=llm,
        tools=[lookup],
        checkpointer=None,
        fallback_llm=fallback_llm,
        llm_idle_timeout=15,
        llm_run_timeout=60,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(USER_TEXT)]})
    return result["messages"][-1].content


async def test_a_deep_graph_turn_sends_one_jsc_request_then_stays_on_the_fallback():
    health = _health(_clock())
    jsc = _Upstream(_jsc_429())
    mistral = _Upstream(_agent_turn)

    answer = await _run_deep_turn(
        _jsc_client(jsc, health), _client(ChatOpenAI, mistral, "https://fallback.example/v1")
    )

    assert answer == "The answer."
    assert jsc.calls == 1
    assert mistral.calls == 2
    assert health.is_rate_limited(JSC) is True


async def test_a_jsc_timeout_in_a_turn_keeps_todays_retry_and_opens_no_window():
    health = _health(_clock())

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    class _TimingOut(_Upstream):
        def __call__(self, request):
            self.calls += 1
            return timeout(request)

    jsc = _TimingOut(None)
    mistral = _Upstream(_agent_turn)

    answer = await _run_react_turn(
        _jsc_client(jsc, health), _client(ChatOpenAI, mistral, "https://fallback.example/v1")
    )

    assert answer == "The answer."
    # As before: each step's timeout routes to the fallback node, and a
    # timeout opens no rate limit window.
    assert jsc.calls == 2
    assert health.is_rate_limited(JSC) is False
    assert health.is_open(JSC) is False


async def test_a_turn_on_another_worker_stays_off_jsc_while_the_probe_is_out(monkeypatch):
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    jsc_b = _Upstream(_agent_turn)
    mistral = _Upstream(_agent_turn)
    fallback = _client(ChatOpenAI, mistral, "https://fallback.example/v1")

    clock.now = 1060.0
    permit = worker_a.admit(JSC)  # worker A's turn is probing

    # Worker B's turn started before the window ended and holds the JSC client.
    answer = await _run_react_turn(_jsc_client(jsc_b, worker_b), fallback)
    assert answer == "The answer."
    assert jsc_b.calls == 0

    worker_a.close_rate_limit_window(JSC, permit.started_at)
    clock.now = 1062.0
    answer = await _run_react_turn(_jsc_client(jsc_b, worker_b), fallback)
    assert jsc_b.calls == 2


async def test_no_log_record_carries_the_key_or_the_user_text(caplog):
    health = _health(_clock())
    jsc = _Upstream(_jsc_429())
    mistral = _Upstream(_agent_turn)
    formatter = logging.Formatter("%(name)s %(message)s")

    with caplog.at_level(logging.INFO):
        await _run_react_turn(
            _jsc_client(jsc, health), _client(ChatOpenAI, mistral, "https://fallback.example/v1")
        )

    rendered = [formatter.format(record) for record in caplog.records]
    assert any("rate limited (HTTP 429)" in line for line in rendered)
    for line in rendered:
        assert SECRET_KEY not in line
        assert USER_TEXT not in line


# ─── backoff ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("count", "retry_after_s", "window"),
    [
        (1, 60, 60),
        (2, 60, 120),
        (3, 30, 240),
        (6, 60, 1920),
        (7, 60, 3600),
        (1, 600, 600),
        (2, 600, 600),
        (1000, 1, 3600),
    ],
)
def test_the_window_doubles_with_each_429_in_a_row(count, retry_after_s, window):
    assert rate_limit_backoff_s(count, retry_after_s) == window


@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
def test_a_probe_that_meets_the_block_doubles_the_window_until_one_answers(shared):
    clock = _clock()
    store = _FakeRedis(clock) if shared else None
    health = _health(clock, store)

    # Blablador keeps answering Retry-After: 60 while it doubles its block.
    start = 1000.0
    for window in (60, 120, 240):
        clock.now = start
        assert health.record_failure(JSC, _rate_limited({"retry-after": "60"})) == window
        clock.now = start + window - 0.5
        with pytest.raises(EndpointRateLimited):
            health.admit(JSC)
        start += window
        clock.now = start
        permit = health.admit(JSC)
        assert permit.probe is True

    # The probe answered: the count starts over.
    assert health.close_rate_limit_window(JSC, permit.started_at) is True
    assert health.record_failure(JSC, _rate_limited({"retry-after": "60"})) == 60


def test_workers_share_the_count_of_429s():
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b, worker_c = (_health(clock, store) for _ in range(3))

    # Requests in flight when the block started each get a 429.
    assert worker_a.record_failure(JSC, _rate_limited({"retry-after": "60"})) == 60
    assert worker_b.record_failure(JSC, _rate_limited({"retry-after": "60"})) == 120
    assert worker_c.record_failure(JSC, _rate_limited({"retry-after": "60"})) == 240

    assert store.peek(KEY + ":rate-limit") == "1000000:1240000:3"
    clock.now = 1239.0
    with pytest.raises(EndpointRateLimited):
        worker_a.admit(JSC)


@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
def test_one_warning_when_the_window_opens_or_grows_none_otherwise(shared, caplog):
    clock = _clock()
    store = _FakeRedis(clock) if shared else None
    worker_a = _health(clock, store)
    worker_b = _health(clock, store) if shared else worker_a

    with caplog.at_level(logging.WARNING, logger="src.core.llm_health"):
        worker_a.record_failure(JSC, _rate_limited({"retry-after": "600"}))
        # Second in a row: 120 s from now ends before the running window.
        clock.now = 1010.0
        worker_b.record_failure(JSC, _rate_limited({"retry-after": "30"}))
        # Third: 240 s from now still ends inside it.
        worker_b.record_failure(JSC, _rate_limited({}))

    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert lines == ["Endpoint eve_jsc circuit opened: rate limited (HTTP 429), retry after 600s"]


# ─── the probe lease ──────────────────────────────────────────────────────────


class _HookedRedis(_FakeRedis):
    """Runs *hook* once, right after the next GET returns."""

    hook = None

    def get(self, key: str):
        value = super().get(key)
        if self.hook is not None:
            hook, self.hook = self.hook, None
            hook()
        return value


def test_a_429_recorded_between_a_read_and_the_lease_keeps_the_other_worker_out():
    clock = _clock()
    store = _HookedRedis(clock)
    worker_a, worker_b = _health(clock, store), _health(clock, store)
    worker_b.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    clock.now = 1061.0
    assert worker_b.admit(JSC).probe is True

    # Worker A reads the ended window; before it asks for the lease, worker
    # B's probe gets a 429 and records it.
    clock.now = 1061.2
    store.hook = lambda: worker_b.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    with pytest.raises(EndpointRateLimited):
        worker_a.admit(JSC)
    assert _remaining(worker_b) == pytest.approx(120.0)


def test_a_window_written_back_after_an_outage_keeps_the_probe_lease():
    from src.core.llm_health import _write_window

    clock = _clock()
    store = _FakeRedis(clock)
    window_key, lease_key = KEY + ":rate-limit", KEY + ":rate-limit:probe"
    store.set(lease_key, "lease-b", px=30_000)

    stored, grew = _write_window(store, window_key, 1_000_000, 1_000_000, until_ms=1_600_000, count=1)

    assert (stored.until_ms, stored.count, grew) == (1_600_000, 1, True)
    assert store.peek(lease_key) == "lease-b"


@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
def test_a_released_probe_lets_the_next_request_probe_at_once(shared):
    clock = _clock()
    store = _FakeRedis(clock) if shared else None
    health = _health(clock, store)
    health.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    clock.now = 1060.0
    permit = health.admit(JSC)
    assert permit.probe is True

    health.release_probe(JSC, permit)

    assert health.admit(JSC).probe is True


def test_releasing_an_expired_lease_leaves_the_new_holder_alone():
    clock = _clock()
    store = _FakeRedis(clock)
    worker_a, worker_b, worker_c = (_health(clock, store) for _ in range(3))
    worker_a.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    clock.now = 1060.0
    stale = worker_a.admit(JSC)
    clock.now = 1060.0 + PROBE_LEASE_S
    assert worker_b.admit(JSC).probe is True

    worker_a.release_probe(JSC, stale)

    with pytest.raises(EndpointRateLimited):
        worker_c.admit(JSC)


def test_the_snapshot_keeps_an_open_circuit_next_to_a_window():
    health = _health(_clock())
    health.record_failure(JSC, TimeoutError("first token budget"))
    health.record_failure(JSC, _rate_limited({"retry-after": "60"}))

    row = health.snapshot()[JSC]

    assert row["open"] is True and row["rate_limited"] is True
    assert row["failures"] == 2
    assert "429" in row["last_error"] and "TimeoutError" in row["last_error"]


# ─── what closes a window ─────────────────────────────────────────────────────


def _sse(*deltas: dict) -> bytes:
    events = [
        b"data: "
        + json.dumps(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        ).encode()
        + b"\n\n"
        for delta in deltas
    ]
    return b"".join(events) + b"data: [DONE]\n\n"


def _streamed(body: bytes):
    def answer(_body: dict, _calls: int) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return answer


def _window_after(shared: bool):
    """A breaker whose 60 s window just ended, so the next call is the probe."""
    clock = _clock()
    store = _FakeRedis(clock) if shared else None
    health = _health(clock, store)
    health.record_failure(JSC, _rate_limited({"retry-after": "60"}))
    clock.now = 1060.0
    return health, clock


def _still_open(health: EndpointHealth, clock: _Clock) -> bool:
    """True when the window was not closed: after the lease, the next call
    is a probe again."""
    clock.now = 1060.0 + PROBE_LEASE_S
    return health.admit(JSC).probe is True


@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
async def test_a_direct_stream_probe_answered_with_a_200_text_plain_keeps_the_window(shared):
    health, clock = _window_after(shared)
    jsc = _Upstream(_jsc_plain_text_200)

    with contextlib.suppress(Exception):
        async for _chunk in _jsc_client(jsc, health).astream("hi"):
            pass

    assert jsc.calls == 1
    assert _still_open(health, clock)


@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
async def test_a_direct_stream_probe_with_no_content_keeps_the_window(shared):
    health, clock = _window_after(shared)
    jsc = _Upstream(_streamed(_sse({"role": "assistant", "content": ""})))

    with contextlib.suppress(Exception):
        async for _chunk in _jsc_client(jsc, health).astream("hi"):
            pass

    assert _still_open(health, clock)


async def test_a_direct_stream_probe_closes_the_window_on_its_first_content():
    health, clock = _window_after(False)
    jsc = _Upstream(_streamed(_sse({"role": "assistant", "content": ""}, {"content": "ok"})))
    client = _jsc_client(jsc, health)

    chunks = [chunk.content async for chunk in client.astream("hi")]

    assert "ok" in chunks
    assert health.is_rate_limited(JSC) is False
    assert health.admit(JSC).probe is False


async def test_an_agentic_streaming_probe_answered_with_a_200_text_plain_keeps_the_window():
    health, clock = _window_after(False)
    jsc = _Upstream(_jsc_plain_text_200)
    mistral = _Upstream(_agent_turn)
    client = _jsc_client(jsc, health)
    client.streaming = True

    answer = await _run_react_turn(client, _client(ChatOpenAI, mistral, "https://fallback.example/v1"))

    assert answer == "The answer."
    assert jsc.calls == 1
    assert _still_open(health, clock)


async def test_a_probe_with_an_empty_reply_keeps_the_window():
    health, clock = _window_after(False)
    jsc = _Upstream(lambda _body, _calls: _completion({"content": ""}))

    await _jsc_client(jsc, health).ainvoke("hi")

    assert _still_open(health, clock)


async def test_a_probe_that_answers_with_a_tool_call_closes_the_window():
    health, _clock_ = _window_after(False)
    jsc = _Upstream(_agent_turn)

    await _jsc_client(jsc, health).ainvoke("hi")

    assert health.admit(JSC).probe is False


# ─── single-shot callers ──────────────────────────────────────────────────────


async def test_a_single_shot_call_refused_by_the_window_answers_from_the_fallback(monkeypatch):
    from src.services import llm_inference

    class _Refusing:
        async def ainvoke(self, _messages):
            raise EndpointRateLimited(JSC, 30)

    class _Fallback:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content="from the fallback")

    manager = SimpleNamespace(
        get_client_for_model=lambda _model: _Refusing(), get_fallback_llm=lambda: _Fallback()
    )
    monkeypatch.setattr(llm_inference, "get_shared_llm_manager", lambda: manager)

    async def _no_budget(*_args, **_kwargs):
        return None

    monkeypatch.setattr(llm_inference, "enforce_token_budget_or_raise", _no_budget)
    monkeypatch.setattr(llm_inference, "consume_tokens_for_user", _no_budget)

    answer = await llm_inference.invoke_llm_and_consume_tokens(
        user=None, model=JSC, messages=[HumanMessage("hi")], prompt_for_token_count="hi"
    )

    assert answer == "from the fallback"
