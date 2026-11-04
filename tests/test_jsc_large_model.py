"""jsc_large: a second model on the JSC endpoint, selectable next to EVE-Instruct."""

from functools import partial
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from langchain_openai import ChatOpenAI
from pymongo.errors import OperationFailure

from src.commands.register_jsc_large_model import (
    CATALOG_ID,
    main as register_main,
    register_jsc_large_model,
)
from src.core import llm_manager as llm_module
from src.core.llm_manager import LLMManager, LLMType, endpoint_timeout
from src.database.models.catalog_platform_model import CatalogPlatformModelDoc
from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest
from src.services import generate_answer
from src.services.agents.core import runner
from src.services.provider_catalog import (
    JSC_PLATFORM_MODEL_ID,
    clear_provider_catalog_cache_for_tests,
)
from tests.test_endpoint_failover import (
    _FakeGraph,
    _new_turn,
    _patch_pipeline,
    _stream,
    _tokens,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

COMMAND = "src.commands.register_jsc_large_model.register_jsc_large_model"
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


def _use_manager(monkeypatch, manager):
    monkeypatch.setattr(generate_answer, "get_shared_llm_manager", lambda: manager)


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
    _use_manager(monkeypatch, _manager(monkeypatch))
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
    _use_manager(monkeypatch, _manager(monkeypatch))
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


async def test_a_blank_alias_hides_the_row_so_no_pick_answers_in_silence(
    async_client, clean_row, monkeypatch
):
    _use_manager(monkeypatch, _manager(monkeypatch, alias=""))
    user, token = await create_test_user_and_token()
    try:
        await register_jsc_large_model(apply=True)
        assert "jsc_large" not in [
            m["llm_type"] for m in await _platform(async_client, token)
        ]
    finally:
        await cleanup_models([user])


async def test_the_transparent_default_is_listed_whatever_the_chain_says(
    async_client, clean_row, monkeypatch
):
    # Staging and prod: the "main" row answers through JSC, whose base URL is what
    # made it transparent. A blank key must not take the only EVE entry away.
    monkeypatch.setattr("src.services.provider_catalog.FEATURE_JSC_MODEL", False)
    monkeypatch.setattr("src.services.provider_catalog.EVE_JSC_BASE_URL", JSC_URL)
    manager = _manager(monkeypatch, key="")
    monkeypatch.setattr(manager, "_is_configured", lambda name: name == "fallback")
    _use_manager(monkeypatch, manager)
    user, token = await create_test_user_and_token()
    try:
        by_type = {m["llm_type"]: m for m in await _platform(async_client, token)}
        assert by_type["eve_jsc"]["display_name"] == "EVE-Instruct"
        assert "fallback" in by_type
    finally:
        await cleanup_models([user])


async def test_listing_models_survives_a_manager_that_cannot_be_built(
    async_client, clean_row, monkeypatch
):
    def broken():
        raise RuntimeError("no manager")

    _use_manager(monkeypatch, None)
    monkeypatch.setattr(generate_answer, "get_shared_llm_manager", broken)
    user, token = await create_test_user_and_token()
    try:
        await register_jsc_large_model(apply=True)
        assert "jsc_large" in [
            m["llm_type"] for m in await _platform(async_client, token)
        ]
    finally:
        await cleanup_models([user])


async def test_the_alias_is_the_model_in_the_request_body(monkeypatch):
    sent = []

    async def respond(request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": ALIAS,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi"},
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        llm_module, "ChatOpenAI", partial(ChatOpenAI, http_async_client=client)
    )
    llm = _manager(monkeypatch).client_for_candidate("jsc_large")

    await llm.ainvoke("hello")

    assert len(sent) == 1
    assert str(sent[0].url) == f"{JSC_URL}/chat/completions"
    assert b'"model":"' + ALIAS.encode() + b'"' in sent[0].content.replace(b" ", b"")


def _rate_limited() -> openai.RateLimitError:
    request = httpx.Request("POST", f"{JSC_URL}/chat/completions")
    return openai.RateLimitError(
        "429", response=httpx.Response(429, request=request), body=None
    )


@pytest.mark.parametrize(
    "failure",
    [_rate_limited(), httpx.ReadTimeout("slow")],
    ids=["429", "timeout"],
)
async def test_a_failure_marks_only_jsc_large_and_the_stream_falls_through(
    monkeypatch, failure
):
    user, _ = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _new_turn(user)
        graph = _FakeGraph({"jsc_large": [failure]})
        manager = _patch_pipeline(
            monkeypatch, graph, configured=("eve_jsc", "jsc_large", "main", "fallback")
        )

        events = await _stream(conversation, message, llm_type="jsc_large")

        assert _tokens(events) == ["fallback answer"]
        assert graph.seen == ["jsc_large"]
        assert manager.health.is_open("jsc_large") is True
        assert manager.health.is_open("eve_jsc") is False
        endpoint = (await Message.find_by_id(message.id)).metadata["endpoint"]
        assert endpoint["answered"] == "fallback"
        assert endpoint["substituted"] is True
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


async def test_apply_with_the_reader_says_so(monkeypatch, capsys):
    denied = OperationFailure("Authorization failure", code=13)
    monkeypatch.setattr(COMMAND, AsyncMock(side_effect=denied))

    assert await register_main(apply=True) == 1
    assert "needs the writer" in capsys.readouterr().out


async def test_other_mongo_errors_are_not_reported_as_permissions(monkeypatch):
    other = OperationFailure("Something else", code=11600)
    monkeypatch.setattr(COMMAND, AsyncMock(side_effect=other))

    with pytest.raises(OperationFailure):
        await register_main(apply=True)
