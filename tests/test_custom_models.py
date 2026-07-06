from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.database.models.user_custom_model import UserCustomModel
from src.services.agents.core.runner import _resolve_agentic_llm_client
from src.services.custom_model_secrets import clear_secret_cache_for_tests
from src.services.generate_answer import GenerationRequest
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


@pytest.fixture(autouse=True)
def _clear_secret_cache():
    clear_secret_cache_for_tests()
    yield
    clear_secret_cache_for_tests()


@pytest.mark.asyncio
async def test_list_models_includes_platform_and_custom(async_client):
    user, token = await create_test_user_and_token()
    try:
        with patch(
            "src.routers.custom_model.create_custom_model_secret",
            new=AsyncMock(return_value="arn:aws:secretsmanager:eu-central-1:123:secret:test"),
        ):
            create_resp = await async_client.post(
                "/users/custom-models",
                json={
                    "display_name": "My GPT",
                    "model_name": "gpt-4o",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "sk-test-key",
                },
                headers={"Authorization": f"Bearer {token}"},
            )
        assert create_resp.status_code == 201
        body = create_resp.json()
        assert "api_key" not in body
        assert body["has_api_key"] is True

        list_resp = await async_client.get(
            "/models",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert any(m["id"] == "eve-instruct" for m in payload["platform"])
        assert len(payload["custom"]) == 1
        assert payload["custom"][0]["display_name"] == "My GPT"
    finally:
        await UserCustomModel.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_create_custom_model_requires_auth(async_client):
    response = await async_client.post(
        "/users/custom-models",
        json={
            "display_name": "My GPT",
            "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test-key",
        },
    )
    assert response.status_code == 403


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
                json={
                    "display_name": "My GPT",
                    "model_name": "gpt-4o",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "sk-test-key",
                },
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
                json={
                    "display_name": "Owner model",
                    "model_name": "gpt-4o",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "sk-test-key",
                },
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
async def test_resolve_agentic_llm_client_uses_custom_model():
    user, _ = await create_test_user_and_token()
    model = await UserCustomModel.create(
        user_id=user.id,
        display_name="Custom",
        model_name="gpt-4o-mini",
        base_url="https://api.example.com/v1",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:abc",
    )
    try:
        request = GenerationRequest(
            query="hello",
            custom_model_id=model.id,
        )
        with patch(
            "src.services.agents.core.runner.get_custom_model_api_key",
            new=AsyncMock(return_value="sk-custom"),
        ), patch(
            "src.services.agents.core.runner.get_shared_llm_manager"
        ) as manager_factory:
            manager = manager_factory.return_value
            manager.build_custom_client.return_value = object()

            llm, prompts = await _resolve_agentic_llm_client(
                request, user_id=user.id
            )

            assert llm is manager.build_custom_client.return_value
            manager.build_custom_client.assert_called_once_with(
                base_url=model.base_url,
                model_name=model.model_name,
                api_key="sk-custom",
            )
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
        model_name="gpt-4o-mini",
        base_url="https://api.example.com/v1",
        secret_arn="arn:aws:secretsmanager:eu-central-1:123:secret:abc",
    )
    try:
        request = GenerationRequest(
            query="hello",
            llm_type="fallback",
            custom_model_id=model.id,
        )
        with patch(
            "src.services.agents.core.runner.get_custom_model_api_key",
            new=AsyncMock(return_value="sk-custom"),
        ), patch(
            "src.services.agents.core.runner.get_shared_llm_manager"
        ) as manager_factory:
            manager = manager_factory.return_value
            manager.build_custom_client.return_value = object()

            _, prompts = await _resolve_agentic_llm_client(request, user_id=user.id)

            manager.get_client_for_model.assert_not_called()
            assert prompts["custom_model_id"] == model.id
    finally:
        await UserCustomModel.delete_many({"user_id": user.id})
        await cleanup_models([user])

    owner, _ = await create_test_user_and_token()
    other, _ = await create_test_user_and_token()
    model = await UserCustomModel.create(
        user_id=owner.id,
        display_name="Owner only",
        model_name="gpt-4o-mini",
        base_url="https://api.example.com/v1",
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
