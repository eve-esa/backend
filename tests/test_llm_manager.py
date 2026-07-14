import pytest

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
