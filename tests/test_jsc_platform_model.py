"""The JSC platform model is injected at serve time, never seeded.

FEATURE_JSC_MODEL controls both visibility and position: on, the entry is
prepended to /models so the frontend, which defaults to the first platform
model, selects it. A blank EVE_JSC_BASE_URL wins over the flag, and a Mongo
document with the same identity wins over the injection.
"""

import pytest

from src.database.models.catalog_platform_model import CatalogPlatformModelDoc
from src.services.provider_catalog import (
    JSC_LLM_TYPE,
    JSC_PLATFORM_MODEL_ID,
    clear_provider_catalog_cache_for_tests,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


@pytest.fixture(autouse=True)
def _clear_catalog_cache():
    clear_provider_catalog_cache_for_tests()
    yield
    clear_provider_catalog_cache_for_tests()


async def _list_platform(async_client, token):
    resp = await async_client.get(
        "/models", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    return resp.json()["platform"]


@pytest.mark.asyncio
async def test_flag_off_hides_the_jsc_model(async_client, monkeypatch):
    monkeypatch.setattr("src.services.provider_catalog.FEATURE_JSC_MODEL", False)
    monkeypatch.setattr(
        "src.services.provider_catalog.EVE_JSC_BASE_URL", "https://jsc.example/v1"
    )
    user, token = await create_test_user_and_token()
    try:
        platform = await _list_platform(async_client, token)
        assert all(m["id"] != JSC_PLATFORM_MODEL_ID for m in platform)
        assert all(m["llm_type"] != JSC_LLM_TYPE for m in platform)
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_flag_on_lists_the_jsc_model_first(async_client, monkeypatch):
    monkeypatch.setattr("src.services.provider_catalog.FEATURE_JSC_MODEL", True)
    monkeypatch.setattr(
        "src.services.provider_catalog.EVE_JSC_BASE_URL", "https://jsc.example/v1"
    )
    user, token = await create_test_user_and_token()
    try:
        platform = await _list_platform(async_client, token)
        assert platform[0]["id"] == JSC_PLATFORM_MODEL_ID
        assert platform[0]["llm_type"] == JSC_LLM_TYPE
        assert platform[0]["display_name"] == "EVE-Instruct (JSC)"
        # The seeded catalog still follows, untouched.
        assert any(m["id"] == "eve-instruct" for m in platform[1:])
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_blank_base_url_wins_over_the_flag(async_client, monkeypatch):
    monkeypatch.setattr("src.services.provider_catalog.FEATURE_JSC_MODEL", True)
    monkeypatch.setattr("src.services.provider_catalog.EVE_JSC_BASE_URL", "")
    user, token = await create_test_user_and_token()
    try:
        platform = await _list_platform(async_client, token)
        assert all(m["id"] != JSC_PLATFORM_MODEL_ID for m in platform)
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_a_mongo_document_wins_over_the_injection(async_client, monkeypatch):
    monkeypatch.setattr("src.services.provider_catalog.FEATURE_JSC_MODEL", True)
    monkeypatch.setattr(
        "src.services.provider_catalog.EVE_JSC_BASE_URL", "https://jsc.example/v1"
    )
    doc = CatalogPlatformModelDoc(
        catalog_id="eve-instruct-jsc-db",
        llm_type=JSC_LLM_TYPE,
        display_name="EVE-Instruct (JSC) from Mongo",
        enabled=True,
        sort_order=99,
    )
    await doc.save()
    clear_provider_catalog_cache_for_tests()
    user, token = await create_test_user_and_token()
    try:
        platform = await _list_platform(async_client, token)
        jsc_entries = [m for m in platform if m["llm_type"] == JSC_LLM_TYPE]
        assert len(jsc_entries) == 1
        assert jsc_entries[0]["id"] == "eve-instruct-jsc-db"
    finally:
        await CatalogPlatformModelDoc.delete_many({"catalog_id": "eve-instruct-jsc-db"})
        await cleanup_models([user])
