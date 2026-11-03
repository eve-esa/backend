"""jsc_large: a second model on the JSC endpoint, selectable next to EVE-Instruct."""

import pytest

from src.commands.register_jsc_large_model import (
    CATALOG_ID,
    register_jsc_large_model,
)
from src.core import llm_manager as llm_module
from src.core.llm_manager import LLMManager, LLMType, endpoint_timeout
from src.database.models.catalog_platform_model import CatalogPlatformModelDoc
from src.schemas.generation_request import GenerationRequest
from src.services import generate_answer
from src.services.agents.core import runner
from src.services.provider_catalog import (
    JSC_PLATFORM_MODEL_ID,
    clear_provider_catalog_cache_for_tests,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

ALIAS = "alias-test-large"
JSC_URL = "https://jsc.example.invalid/v1"


def _manager(monkeypatch, *, alias=ALIAS, base_url=JSC_URL, key="jsc-key"):
    monkeypatch.setattr(llm_module, "EVE_JSC_BASE_URL", base_url)
    monkeypatch.setattr(llm_module, "EVE_JSC_API_KEY", key)
    monkeypatch.setattr(llm_module, "EVE_JSC_MODEL_NAME", "alias-eve")
    monkeypatch.setattr(llm_module, "JSC_LARGE_MODEL_NAME", alias)
    monkeypatch.setattr(llm_module, "JSC_LARGE_TIMEOUT", 77)
    return LLMManager()


@pytest.fixture(autouse=True)
def _clear_catalog_cache():
    clear_provider_catalog_cache_for_tests()
    yield
    clear_provider_catalog_cache_for_tests()


@pytest.fixture
async def clean_row():
    await CatalogPlatformModelDoc.delete_many({"catalog_id": CATALOG_ID})
    yield
    await CatalogPlatformModelDoc.delete_many({"catalog_id": CATALOG_ID})


def test_an_explicit_request_leads_with_it_and_falls_back(monkeypatch):
    manager = _manager(monkeypatch)
    assert manager.resolve_chain("jsc_large") == ["jsc_large", "fallback"]


@pytest.mark.parametrize(
    "overrides",
    [{"alias": ""}, {"base_url": ""}, {"key": ""}],
    ids=["no alias", "no base url", "no key"],
)
def test_a_missing_setting_drops_the_type_and_fallback_answers(monkeypatch, overrides):
    manager = _manager(monkeypatch, **overrides)
    assert manager.resolve_chain("jsc_large") == ["fallback"]


def test_the_default_chain_stays_without_it(monkeypatch):
    manager = _manager(monkeypatch)
    monkeypatch.setattr(llm_module, "EVE_ENDPOINT_ORDER", "eve_jsc,main,fallback")
    assert "jsc_large" not in manager.resolve_chain(None)


def test_the_client_sends_the_alias_with_its_own_timeout(monkeypatch):
    client = _manager(monkeypatch).client_for_candidate("jsc_large")
    assert client.model_name == ALIAS
    assert str(client.openai_api_base).rstrip("/") == JSC_URL
    assert client.request_timeout == 77


def test_eve_instruct_keeps_its_model(monkeypatch):
    manager = _manager(monkeypatch)
    manager.client_for_candidate("jsc_large")
    assert manager.client_for_candidate("eve_jsc").model_name == "alias-eve"


def test_timeout_and_model_name_are_its_own(monkeypatch):
    monkeypatch.setattr(llm_module, "JSC_LARGE_TIMEOUT", 77)
    monkeypatch.setattr(llm_module, "EVE_JSC_TIMEOUT", 15)
    assert endpoint_timeout("jsc_large") == 77
    assert endpoint_timeout("eve_jsc") == 15
    monkeypatch.setattr(generate_answer, "JSC_LARGE_MODEL_NAME", ALIAS)
    assert generate_answer._resolve_model_name_for_llm_type("jsc_large") == ALIAS
    monkeypatch.setattr(generate_answer, "JSC_LARGE_MODEL_NAME", "")
    assert generate_answer._resolve_model_name_for_llm_type("jsc_large") is None


async def test_an_agentic_turn_sends_the_alias_and_names_it(monkeypatch):
    monkeypatch.setattr(runner, "get_shared_llm_manager", lambda: manager)
    manager = _manager(monkeypatch)
    llm, metadata = await runner._resolve_agentic_llm_client(
        GenerationRequest(query="q", llm_type="jsc_large"), user_id="u"
    )
    assert llm.model_name == ALIAS
    assert metadata["endpoint"]["answered"] == "jsc_large"
    assert metadata["endpoint"]["substituted"] is False


async def test_a_blank_alias_answers_with_fallback_and_says_so(monkeypatch):
    monkeypatch.setattr(runner, "get_shared_llm_manager", lambda: manager)
    manager = _manager(monkeypatch, alias="")
    _, metadata = await runner._resolve_agentic_llm_client(
        GenerationRequest(query="q", llm_type="jsc_large"), user_id="u"
    )
    assert metadata["endpoint"]["answered"] == "fallback"
    assert metadata["endpoint"]["substituted"] is True


async def _platform(async_client, token):
    clear_provider_catalog_cache_for_tests()
    resp = await async_client.get(
        "/models", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    return resp.json()["platform"]


async def test_the_command_is_a_dry_run_by_default(clean_row, capsys):
    await register_jsc_large_model()
    assert "Dry run" in capsys.readouterr().out
    assert await CatalogPlatformModelDoc.count_documents({"catalog_id": CATALOG_ID}) == 0


async def test_the_row_lists_it_last_and_disable_hides_it(
    async_client, clean_row, monkeypatch
):
    monkeypatch.setattr("src.services.provider_catalog.FEATURE_JSC_MODEL", True)
    monkeypatch.setattr("src.services.provider_catalog.EVE_JSC_BASE_URL", JSC_URL)
    user, token = await create_test_user_and_token()
    try:
        await register_jsc_large_model(apply=True)
        await register_jsc_large_model(apply=True)  # re-running adds nothing
        assert await CatalogPlatformModelDoc.count_documents({"catalog_id": CATALOG_ID}) == 1

        platform = await _platform(async_client, token)
        assert platform[-1]["llm_type"] == "jsc_large"
        assert platform[-1]["display_name"] == "Mistral Large 4"
        # It must not read as the EVE-Instruct JSC entry: the default stays first.
        assert platform[0]["id"] == JSC_PLATFORM_MODEL_ID
        assert platform[0]["llm_type"] == "eve_jsc"

        await register_jsc_large_model(apply=True, disable=True)
        assert "jsc_large" not in [
            m["llm_type"] for m in await _platform(async_client, token)
        ]
    finally:
        await cleanup_models([user])


async def test_the_transparent_default_is_untouched(async_client, clean_row, monkeypatch):
    # Staging and prod shape: flag off, JSC configured, the EVE pick goes to JSC.
    monkeypatch.setattr("src.services.provider_catalog.FEATURE_JSC_MODEL", False)
    monkeypatch.setattr("src.services.provider_catalog.EVE_JSC_BASE_URL", JSC_URL)
    user, token = await create_test_user_and_token()
    try:
        await register_jsc_large_model(apply=True)
        by_type = {m["llm_type"]: m for m in await _platform(async_client, token)}
        assert by_type["eve_jsc"]["display_name"] == "EVE-Instruct"
        assert by_type["jsc_large"]["display_name"] == "Mistral Large 4"
    finally:
        await cleanup_models([user])


def test_the_enum_value_is_stable():
    assert LLMType.Jsc_Large.value == "jsc_large"
