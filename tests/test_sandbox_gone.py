"""A sandbox OpenSandbox no longer knows is expected, not a warning.

These tests run the real OpenSandbox SDK against an httpx mock transport, so
the SDK's own log line is the one the HyperDX gate saw.
"""

import logging
from datetime import timedelta

import httpx
import pytest

opensandbox_config = pytest.importorskip("opensandbox.config")

from src.services.agents.core import sandbox as sandbox_module  # noqa: E402
from src.services.agents.core.sandbox import (  # noqa: E402
    SandboxGone,
    _attach_opensandbox,
    kill_record,
)

pytestmark = pytest.mark.no_db

_SDK_LOGGER = "opensandbox.sync.adapters.sandboxes_adapter"
_RECORD = {"provider": "opensandbox", "sandbox_id": "sb-gone", "skills_digest": ""}


def _use_transport(monkeypatch, handler):
    """Make ``_opensandbox_connection`` build its config on a mock transport."""
    real = opensandbox_config.ConnectionConfigSync

    def config(**kwargs):
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(opensandbox_config, "ConnectionConfigSync", config)
    monkeypatch.setenv("SANDBOX_OPENSANDBOX_DOMAIN", "sandbox.internal")
    monkeypatch.delenv("SANDBOX_OPENSANDBOX_PROTOCOL", raising=False)
    monkeypatch.delenv("SANDBOX_OPENSANDBOX_API_KEY", raising=False)


def _not_found(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        404,
        json={"code": "SANDBOX::NOT_FOUND", "message": "Sandbox sb-gone not found"},
    )


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_attach_to_a_gone_sandbox_raises_gone_without_a_warning(monkeypatch, caplog):
    _use_transport(monkeypatch, _not_found)

    with caplog.at_level(logging.DEBUG), pytest.raises(SandboxGone):
        _attach_opensandbox(dict(_RECORD))

    assert _warnings(caplog) == []


def test_kill_of_a_gone_sandbox_logs_info_with_the_id_only(monkeypatch, caplog):
    _use_transport(monkeypatch, _not_found)

    with caplog.at_level(logging.DEBUG):
        kill_record(dict(_RECORD))

    assert _warnings(caplog) == []
    gone = [r for r in caplog.records if r.getMessage().startswith("sandbox.already_gone")]
    assert [r.getMessage() for r in gone] == [
        "sandbox.already_gone provider=opensandbox sandbox_id=sb-gone"
    ]
    assert gone[0].levelname == "INFO"


def test_attach_timeout_keeps_the_sdk_warning_and_is_not_gone(monkeypatch, caplog):
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    _use_transport(monkeypatch, timeout)

    with caplog.at_level(logging.DEBUG), pytest.raises(Exception) as caught:
        _attach_opensandbox(dict(_RECORD))

    assert not isinstance(caught.value, SandboxGone)
    warned = [r for r in _warnings(caplog) if r.name == _SDK_LOGGER]
    assert len(warned) == 1
    assert warned[0].getMessage().startswith("Failed to retrieve sandbox endpoint")


@pytest.mark.parametrize("status", [401, 500, 503])
def test_attach_server_or_auth_error_keeps_the_sdk_warning(monkeypatch, caplog, status):
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"code": "X", "message": "upstream broke"})

    _use_transport(monkeypatch, failing)

    with caplog.at_level(logging.DEBUG), pytest.raises(Exception) as caught:
        _attach_opensandbox(dict(_RECORD))

    assert not isinstance(caught.value, SandboxGone)
    assert [r.name for r in _warnings(caplog)] == [
        _SDK_LOGGER,
        "src.services.agents.core.sandbox",
    ]


def test_a_not_found_line_outside_attach_keeps_its_warning(caplog):
    with caplog.at_level(logging.DEBUG):
        logging.getLogger(_SDK_LOGGER).warning(
            "Failed to retrieve sandbox endpoint for sandbox sb-x: "
            "Get endpoint for sandbox sb-x port 44772 failed: Sandbox sb-x not found"
        )

    assert [r.levelname for r in caplog.records if r.name == _SDK_LOGGER] == ["WARNING"]


# Readiness retries: the SDK transport logs every retry at WARNING while execd
# starts. Inside our create and connect a retry is expected and only retries
# that run out are a warning; elsewhere a retry keeps its WARNING.

_RETRY_LOGGER = "opensandbox.transport._sync_retry"


def _retrying_transport(handler, max_retries=3):
    from opensandbox.transport import RetryPolicy, RetrySyncTransport

    policy = RetryPolicy(
        max_retries=max_retries,
        initial_backoff=timedelta(milliseconds=1),
        max_backoff=timedelta(milliseconds=2),
    )
    return RetrySyncTransport(httpx.MockTransport(handler), policy)


def test_a_readiness_retry_that_recovers_logs_info_not_warning(caplog):
    # Inside our create or connect call, where the SDK runs the readiness ping.
    quiet = sandbox_module._quiet_retries.set(True)
    try:
        _flaky_get(caplog)
    finally:
        sandbox_module._quiet_retries.reset(quiet)

    retried = [r for r in caplog.records if r.name == _RETRY_LOGGER]
    assert len(retried) == 1
    assert retried[0].getMessage().startswith("retrying GET ")
    assert retried[0].levelname == "INFO"
    assert _warnings(caplog) == []


def test_a_retry_inside_a_tool_call_keeps_its_warning(caplog):
    # A command run or an upload mid-turn: an outage there must leave a WARNING.
    _flaky_get(caplog)

    retried = [r for r in caplog.records if r.name == _RETRY_LOGGER]
    assert [r.levelname for r in retried] == ["WARNING"]


def test_a_reloaded_module_replaces_its_filters():
    logger_name = "eve.test.sandbox_filters"
    target = logging.getLogger(logger_name)

    class _RetryAsInfoFilter(logging.Filter):  # the class from an older load
        pass

    old = _RetryAsInfoFilter()
    setattr(old, sandbox_module._FILTER_MARK, "_RetryAsInfoFilter")
    target.addFilter(old)
    unrelated = logging.Filter()
    target.addFilter(unrelated)
    try:
        new = sandbox_module._RetryAsInfoFilter()
        sandbox_module._add_filter_once(logger_name, new)
        sandbox_module._add_filter_once(logger_name, sandbox_module._RetryAsInfoFilter())

        ours = [f for f in target.filters if hasattr(f, sandbox_module._FILTER_MARK)]
        assert len(ours) == 1 and old not in target.filters
        assert unrelated in target.filters
    finally:
        for f in list(target.filters):
            target.removeFilter(f)


def _flaky_get(caplog):
    calls = {"n": 0}

    def flaky_ping(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.RemoteProtocolError("unexpected eof", request=request)
        return httpx.Response(200)

    with caplog.at_level(logging.DEBUG):
        with httpx.Client(transport=_retrying_transport(flaky_ping)) as client:
            assert client.get("http://host.docker.internal:1/proxy/44772/ping").status_code == 200


def test_retries_that_run_out_log_one_warning_with_the_id(monkeypatch, caplog):
    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"code": "X", "message": "starting"})

    real = opensandbox_config.ConnectionConfigSync

    def config(**kwargs):
        return real(transport=_retrying_transport(unavailable, max_retries=2), **kwargs)

    monkeypatch.setattr(opensandbox_config, "ConnectionConfigSync", config)
    monkeypatch.setenv("SANDBOX_OPENSANDBOX_DOMAIN", "sandbox.internal")

    with caplog.at_level(logging.DEBUG), pytest.raises(Exception) as caught:
        _attach_opensandbox(dict(_RECORD))

    assert not isinstance(caught.value, SandboxGone)
    retried = [r for r in caplog.records if r.name == _RETRY_LOGGER]
    assert len(retried) == 2
    assert {r.levelname for r in retried} == {"INFO"}
    ours = [r for r in _warnings(caplog) if r.getMessage().startswith("sandbox.connect_failed")]
    assert [r.getMessage() for r in ours] == [
        "sandbox.connect_failed provider=opensandbox sandbox_id=sb-gone "
        "error=SandboxApiException"
    ]
