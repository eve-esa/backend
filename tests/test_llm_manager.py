import pytest

from src.config import EVE_JSC_TIMEOUT, MAIN_MODEL_TIMEOUT
from src.core.llm_manager import LLMManager, LLMType


@pytest.mark.parametrize(
    ("llm_type", "expected"),
    [
        ("main", LLMType.Main.value),
        ("runpod", LLMType.Main.value),
        ("fallback", LLMType.Fallback.value),
        ("mistral", LLMType.Fallback.value),
        ("satcom_small", LLMType.Satcom_Small.value),
        ("satcom_large", LLMType.Satcom_Large.value),
        ("eve_jsc", LLMType.Eve_Jsc.value),
    ],
)
def test_get_client_for_model_sets_selected_type(monkeypatch, llm_type, expected):
    manager = LLMManager()

    monkeypatch.setattr(manager, "_get_main_llm", lambda: object())
    monkeypatch.setattr(manager, "_get_fallback_llm", lambda: object())
    monkeypatch.setattr(manager, "_get_satcom_small_llm", lambda: object())
    monkeypatch.setattr(manager, "_get_satcom_large_llm", lambda: object())
    monkeypatch.setattr(manager, "_get_eve_jsc_llm", lambda: object())

    manager.get_client_for_model(llm_type)
    assert manager.get_selected_llm_type() == expected


def test_build_custom_client_does_not_change_selected_llm_type(monkeypatch):
    manager = LLMManager()
    manager.set_selected_llm_type(LLMType.Main.value)

    monkeypatch.setattr(
        "src.core.llm_manager.ChatOpenAI",
        lambda **kwargs: object(),
    )

    manager.build_custom_client(
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o",
        api_key="sk-test",
    )

    assert manager.get_selected_llm_type() == LLMType.Main.value


def _chain_manager(
    monkeypatch, *, configured, order="eve_jsc,main,fallback"
) -> LLMManager:
    """A manager whose configured endpoints and chain order the test decides."""
    monkeypatch.setattr("src.core.llm_manager.EVE_ENDPOINT_ORDER", order)
    manager = LLMManager()
    monkeypatch.setattr(manager, "_is_configured", lambda name: name in configured)
    return manager


def test_unnamed_request_takes_the_configured_chain(monkeypatch):
    """Regression: prod used to answer every unnamed request with Mistral."""
    manager = _chain_manager(monkeypatch, configured={"eve_jsc", "main", "fallback"})

    assert manager.resolve_chain(None) == ["eve_jsc", "main", "fallback"]


def test_unconfigured_endpoints_drop_out_of_the_chain(monkeypatch):
    manager = _chain_manager(monkeypatch, configured={"main", "fallback"})

    assert manager.resolve_chain(None) == ["main", "fallback"]


def test_the_fallback_model_is_always_last(monkeypatch):
    manager = _chain_manager(
        monkeypatch,
        configured={"eve_jsc", "main", "fallback"},
        order="fallback,eve_jsc,main",
    )

    assert manager.resolve_chain(None) == ["eve_jsc", "main", "fallback"]


def test_unknown_chain_entries_are_dropped(monkeypatch):
    manager = _chain_manager(
        monkeypatch,
        configured={"main", "fallback"},
        order="eve_jsc_typo,main",
    )

    assert manager.resolve_chain(None) == ["main", "fallback"]


def test_a_chain_with_nothing_configured_raises(monkeypatch):
    manager = _chain_manager(monkeypatch, configured=set())

    with pytest.raises(RuntimeError):
        manager.resolve_chain(None)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("eve_jsc", ["eve_jsc", "fallback"]),
        ("main", ["main", "fallback"]),
        ("runpod", ["main", "fallback"]),
        ("fallback", ["fallback"]),
        ("mistral", ["fallback"]),
        ("satcom_small", ["satcom_small", "fallback"]),
        ("satcom_large", ["satcom_large", "fallback"]),
    ],
)
def test_an_explicit_request_is_never_promoted(monkeypatch, requested, expected):
    manager = _chain_manager(
        monkeypatch,
        configured={"eve_jsc", "main", "fallback", "satcom_small", "satcom_large"},
    )

    assert manager.resolve_chain(requested) == expected


def test_an_explicit_unconfigured_request_falls_back(monkeypatch):
    manager = _chain_manager(monkeypatch, configured={"main", "fallback"})

    assert manager.resolve_chain("eve_jsc") == ["fallback"]


def test_an_open_circuit_sinks_to_the_back(monkeypatch):
    manager = _chain_manager(monkeypatch, configured={"eve_jsc", "main", "fallback"})
    manager.health.record_failure("eve_jsc", TimeoutError("cold start"))

    assert manager.resolve_chain(None) == ["main", "eve_jsc", "fallback"]


def test_a_chain_of_open_circuits_keeps_its_order(monkeypatch):
    manager = _chain_manager(monkeypatch, configured={"eve_jsc", "main", "fallback"})
    manager.health.record_failure("eve_jsc", TimeoutError("cold start"))
    manager.health.record_failure("main", TimeoutError("cold start"))

    assert manager.resolve_chain(None) == ["eve_jsc", "main", "fallback"]


def test_client_for_candidate_does_not_substitute(monkeypatch):
    manager = _chain_manager(monkeypatch, configured={"fallback"})
    monkeypatch.setattr(
        manager,
        "_get_eve_jsc_llm",
        lambda: (_ for _ in ()).throw(RuntimeError("EVE_JSC_API_KEY is not set")),
    )

    with pytest.raises(RuntimeError):
        manager.client_for_candidate(LLMType.Eve_Jsc.value)


@pytest.mark.parametrize(
    ("getter", "base_url_attr", "api_key_name", "expected_timeout"),
    [
        ("_get_main_llm", "_main_base_url", "MAIN_MODEL_API_KEY", MAIN_MODEL_TIMEOUT),
        ("_get_eve_jsc_llm", "_eve_jsc_base_url", "EVE_JSC_API_KEY", EVE_JSC_TIMEOUT),
    ],
)
def test_each_endpoint_carries_its_own_first_token_budget(
    monkeypatch, getter, base_url_attr, api_key_name, expected_timeout
):
    captured = {}
    monkeypatch.setattr(
        "src.core.llm_manager.ChatOpenAI", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(f"src.core.llm_manager.{api_key_name}", "test-key")
    manager = LLMManager()
    monkeypatch.setattr(manager, base_url_attr, "https://endpoint.example/v1")

    getattr(manager, getter)()

    assert captured["timeout"] == expected_timeout
    assert captured["max_retries"] == 0
