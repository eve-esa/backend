from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.database.models.user_custom_model import UserCustomModel
from src.schemas.generation_request import GenerationRequest
from src.services.agents.core.runner import _resolve_agentic_llm_client
from src.services.custom_model_secrets import (
    clear_secret_cache_for_tests,
    update_custom_model_secret,
)
from src.services.agentic_utils import is_agentic_generation_request
from src.services.custom_model_service import (
    custom_model_id_from_messages,
    get_owned_custom_model,
)
from src.services.provider_catalog import clear_provider_catalog_cache_for_tests
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

CREATE_PAYLOAD = {
    "display_name": "My OpenAI",
    "provider_id": "openai",
    "catalog_model_id": "gpt-5.4-2026-03-05",
    "api_key": "sk-test-key",
}


@pytest.fixture(autouse=True)
def _clear_secret_cache():
    clear_secret_cache_for_tests()
    clear_provider_catalog_cache_for_tests()
    yield
    clear_secret_cache_for_tests()
    clear_provider_catalog_cache_for_tests()


@pytest.mark.asyncio
async def test_list_models_includes_platform_providers_and_custom(async_client):
    user, token = await create_test_user_and_token()
    try:
        with patch(
            "src.routers.custom_model.create_custom_model_secret",
            new=AsyncMock(return_value="arn:aws:secretsmanager:eu-central-1:123:secret:test"),
        ):
            create_resp = await async_client.post(
                "/users/custom-models",
                json=CREATE_PAYLOAD,
                headers={"Authorization": f"Bearer {token}"},
            )
        assert create_resp.status_code == 201
        body = create_resp.json()
        assert "api_key" not in body
        assert body["has_api_key"] is True
        assert body["provider_id"] == "openai"
        assert "base_url" not in body

        list_resp = await async_client.get(
            "/models",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert any(m["id"] == "eve-instruct" for m in payload["platform"])
        assert any(m["id"] == "mistral-small-latest" for m in payload["platform"])
        assert any(p["id"] == "openai" for p in payload["providers"])
        assert len(payload["custom"]) == 1
        assert payload["custom"][0]["display_name"] == "My OpenAI"
    finally:
        await UserCustomModel.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_list_models_skips_legacy_custom_models(async_client):
    user, token = await create_test_user_and_token()
    try:
        collection = UserCustomModel.get_collection()
        await collection.insert_one(
            {
                "user_id": user.id,
                "display_name": "Legacy model",
                "model_name": "gpt-4",
                "base_url": "https://api.openai.com/v1",
                "secret_arn": "arn:aws:secretsmanager:eu-central-1:123:secret:legacy",
            }
        )

        with patch(
            "src.routers.custom_model.create_custom_model_secret",
            new=AsyncMock(return_value="arn:aws:secretsmanager:eu-central-1:123:secret:test"),
        ):
            create_resp = await async_client.post(
                "/users/custom-models",
                json=CREATE_PAYLOAD,
                headers={"Authorization": f"Bearer {token}"},
            )
        assert create_resp.status_code == 201

        list_resp = await async_client.get(
            "/models",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert len(payload["custom"]) == 1
        assert payload["custom"][0]["display_name"] == "My OpenAI"
    finally:
        await UserCustomModel.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_create_custom_model_requires_auth(async_client):
    response = await async_client.post(
        "/users/custom-models",
        json=CREATE_PAYLOAD,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_custom_model_rejects_unknown_provider(async_client):
    user, token = await create_test_user_and_token()
    try:
        response = await async_client.post(
            "/users/custom-models",
            json={
                **CREATE_PAYLOAD,
                "provider_id": "unknown-provider",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_update_and_delete_custom_model(async_client):
    user, token = await create_test_user_and_token()
    secret_arn = "arn:aws:secretsmanager:eu-central-1:123:secret:test"
    try:
        with patch(
            "src.routers.custom_model.create_custom_model_secret",
            new=AsyncMock(return_value=secret_arn),
        ), patch(
            "src.routers.custom_model.update_custom_model_secret",
            new=AsyncMock(),
        ) as update_secret, patch(
            "src.routers.custom_model.delete_custom_model_secret",
            new=AsyncMock(),
        ) as delete_secret:
            create_resp = await async_client.post(
                "/users/custom-models",
                json=CREATE_PAYLOAD,
                headers={"Authorization": f"Bearer {token}"},
            )
            model_id = create_resp.json()["id"]

            patch_resp = await async_client.patch(
                f"/users/custom-models/{model_id}",
                json={"display_name": "Renamed", "api_key": "sk-new-key"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert patch_resp.status_code == 200
            assert patch_resp.json()["display_name"] == "Renamed"
            update_secret.assert_awaited_once()

            delete_resp = await async_client.delete(
                f"/users/custom-models/{model_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert delete_resp.status_code == 204
            delete_secret.assert_awaited_once_with(secret_arn)

            list_resp = await async_client.get(
                "/models",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert list_resp.json()["custom"] == []
    finally:
        await UserCustomModel.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_delete_custom_model_keeps_row_when_secret_delete_fails(async_client):
    user, token = await create_test_user_and_token()
    secret_arn = "arn:aws:secretsmanager:eu-central-1:123:secret:test"
    try:
        with patch(
            "src.routers.custom_model.create_custom_model_secret",
            new=AsyncMock(return_value=secret_arn),
        ), patch(
            "src.routers.custom_model.delete_custom_model_secret",
            new=AsyncMock(side_effect=RuntimeError("sm down")),
        ):
            create_resp = await async_client.post(
                "/users/custom-models",
                json=CREATE_PAYLOAD,
                headers={"Authorization": f"Bearer {token}"},
            )
            model_id = create_resp.json()["id"]

            delete_resp = await async_client.delete(
                f"/users/custom-models/{model_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert delete_resp.status_code == 500

            list_resp = await async_client.get(
                "/models",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert len(list_resp.json()["custom"]) == 1
            assert list_resp.json()["custom"][0]["id"] == model_id
    finally:
        await UserCustomModel.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_custom_model_ownership_enforced(async_client):
    owner, owner_token = await create_test_user_and_token()
    other, other_token = await create_test_user_and_token()
    try:
        with patch(
            "src.routers.custom_model.create_custom_model_secret",
            new=AsyncMock(return_value="arn:aws:secretsmanager:eu-central-1:123:secret:test"),
        ):
            create_resp = await async_client.post(
                "/users/custom-models",
                json=CREATE_PAYLOAD,
                headers={"Authorization": f"Bearer {owner_token}"},
            )
        model_id = create_resp.json()["id"]

        forbidden = await async_client.patch(
            f"/users/custom-models/{model_id}",
            json={"display_name": "Hijacked"},
            headers={"Authorization": f"Bearer {other_token}"},
        )
        assert forbidden.status_code == 403
    finally:
        await UserCustomModel.delete_many({"user_id": owner.id})
        await UserCustomModel.delete_many({"user_id": other.id})
        await cleanup_models([owner, other])


@pytest.mark.asyncio
async def test_resolve_agentic_llm_client_uses_catalog_base_url():
    user, _ = await create_test_user_and_token()
    model = await UserCustomModel.create(
        user_id=user.id,
        display_name="Custom",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:abc",
    )
    try:
        request = GenerationRequest(
            query="hello",
            custom_model_id=model.id,
        )
        with patch(
            "src.services.agents.core.runner.build_custom_model_llm",
            new=AsyncMock(return_value=object()),
        ) as build_llm:
            llm, prompts = await _resolve_agentic_llm_client(
                request, user_id=user.id
            )

            build_llm.assert_awaited_once()
            assert llm is build_llm.return_value
            assert prompts["custom_model_id"] == model.id
            assert prompts["custom_model_display_name"] == "Custom"
    finally:
        await UserCustomModel.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_resolve_agentic_llm_client_custom_model_overrides_llm_type():
    user, _ = await create_test_user_and_token()
    model = await UserCustomModel.create(
        user_id=user.id,
        display_name="Custom",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:abc",
    )
    try:
        request = GenerationRequest(
            query="hello",
            llm_type="fallback",
            custom_model_id=model.id,
        )
        with patch(
            "src.services.agents.core.runner.build_custom_model_llm",
            new=AsyncMock(return_value=object()),
        ), patch(
            "src.services.agents.core.runner.get_shared_llm_manager"
        ) as manager_factory:
            manager = manager_factory.return_value

            _, prompts = await _resolve_agentic_llm_client(request, user_id=user.id)

            manager.get_client_for_model.assert_not_called()
            assert prompts["custom_model_id"] == model.id
    finally:
        await UserCustomModel.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_resolve_agentic_llm_client_rejects_other_users_model():
    owner, _ = await create_test_user_and_token()
    other, _ = await create_test_user_and_token()
    model = await UserCustomModel.create(
        user_id=owner.id,
        display_name="Owner only",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:abc",
    )
    try:
        request = GenerationRequest(query="hello", custom_model_id=model.id)
        with pytest.raises(HTTPException) as exc:
            await _resolve_agentic_llm_client(request, user_id=other.id)
        assert exc.value.status_code == 403
    finally:
        await UserCustomModel.delete_many({"user_id": owner.id})
        await cleanup_models([owner, other])


@pytest.mark.asyncio
async def test_update_custom_model_secret_calls_put_secret_value_with_secret_id():
    secret_arn = "arn:aws:secretsmanager:eu-central-1:123:secret:test"
    mock_client = MagicMock()
    with patch(
        "src.services.custom_model_secrets._client", return_value=mock_client
    ):
        await update_custom_model_secret(secret_arn=secret_arn, api_key="sk-new")

    mock_client.put_secret_value.assert_called_once_with(
        SecretId=secret_arn,
        SecretString='{"api_key": "sk-new"}',
    )


def test_is_agentic_generation_request_detects_custom_model():
    request = GenerationRequest(query="hi", custom_model_id="model-1")
    assert is_agentic_generation_request(request) is True


def test_is_agentic_generation_request_detects_trace_only():
    request = GenerationRequest(query="hi")
    message = MagicMock(trace=[{"role": "assistant"}])
    assert is_agentic_generation_request(request, message) is True


def test_custom_model_id_from_messages_uses_most_recent():
    older = MagicMock(
        request_input=GenerationRequest(query="a", custom_model_id="old")
    )
    newer = MagicMock(
        request_input=GenerationRequest(query="b", custom_model_id="new")
    )
    assert custom_model_id_from_messages([older, newer]) == "new"


@pytest.mark.asyncio
async def test_get_owned_custom_model_requires_credentials_for_use():
    user, _ = await create_test_user_and_token()
    model = await UserCustomModel.create(
        user_id=user.id,
        display_name="No key",
        provider_id="openai",
        catalog_model_id="gpt-5.4-2026-03-05",
        model_name="gpt-5.4-2026-03-05",
        secret_arn="",
    )
    try:
        with pytest.raises(HTTPException) as exc:
            await get_owned_custom_model(model.id, user.id, action="use")
        assert exc.value.status_code == 422
    finally:
        await UserCustomModel.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_maybe_rollup_uses_custom_summarizer():
    from src.services.generate_answer import maybe_rollup_and_trim_history

    custom_llm = MagicMock()
    request = GenerationRequest(query="hi", custom_model_id="model-1")
    msg = MagicMock()
    msg.input = "hi"
    msg.output = "hello"
    msg.request_input = request

    convo = MagicMock()
    convo.user_id = "user-1"
    convo.summary = None
    convo.save = AsyncMock()

    with patch(
        "src.services.generate_answer.Message.count_documents",
        new=AsyncMock(return_value=2),
    ), patch(
        "src.services.generate_answer.Message.find_all",
        new=AsyncMock(return_value=[msg]),
    ), patch(
        "src.services.generate_answer.Conversation.find_by_id",
        new=AsyncMock(return_value=convo),
    ), patch(
        "src.services.generate_answer.build_custom_model_llm_for_user",
        new=AsyncMock(return_value=custom_llm),
    ), patch(
        "src.services.generate_answer.get_shared_llm_manager"
    ) as manager_factory:
        manager = manager_factory.return_value
        manager.summarize_context_in_all = AsyncMock(return_value="summary")

        await maybe_rollup_and_trim_history("conv-1")

        manager.summarize_context_in_all.assert_awaited_once()
        assert manager.summarize_context_in_all.await_args.kwargs["llm"] is custom_llm
        convo.save.assert_awaited_once()
