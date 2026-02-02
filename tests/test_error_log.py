import pytest

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models
from src.database.models.error_log import ErrorLog


@pytest.mark.asyncio
async def test_log_error_success(async_client):
    """Test successful error logging with required fields."""
    user, token = await create_test_user_and_token()
    try:
        payload = {
            "error_message": "Test error message",
            "error_type": "TypeError",
        }

        response = await async_client.post(
            "/log-error",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "Error logged successfully"
        assert "id" in body

        # Verify error log was created in database
        error_log = await ErrorLog.find_by_id(body["id"])
        assert error_log is not None
        assert error_log.user_id == user.id
        assert error_log.error_type == "TypeError"
        assert error_log.error["message"] == "Test error message"
        assert error_log.logger_name == "frontend"
        assert error_log.component == "FRONTEND"
        assert error_log.pipeline_stage == "CLIENT_ERROR"
        assert error_log.conversation_id is None
        assert error_log.message_id is None

        await cleanup_models([error_log])
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_log_error_with_all_fields(async_client):
    """Test error logging with all optional fields."""
    user, token = await create_test_user_and_token()
    try:
        payload = {
            "error_message": "Complete error test",
            "error_type": "ReferenceError",
            "error_stack": "Error: at line 1\n    at function (file.js:10:5)",
            "url": "https://example.com/page",
            "user_agent": "Mozilla/5.0",
            "component": "CUSTOM_COMPONENT",
            "description": "Custom error description",
            "metadata": {"key1": "value1", "key2": 123},
        }

        response = await async_client.post(
            "/log-error",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert "id" in body

        # Verify all fields were saved correctly
        error_log = await ErrorLog.find_by_id(body["id"])
        assert error_log is not None
        assert error_log.user_id == user.id
        assert error_log.error_type == "ReferenceError"
        assert error_log.error["message"] == "Complete error test"
        assert error_log.error["stack"] == payload["error_stack"]
        assert error_log.error["url"] == payload["url"]
        assert error_log.error["user_agent"] == payload["user_agent"]
        assert error_log.error["metadata"] == payload["metadata"]
        assert error_log.component == "CUSTOM_COMPONENT"
        assert error_log.description == "Custom error description"

        await cleanup_models([error_log])
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_log_error_without_authentication(async_client):
    """Test that error logging requires authentication."""
    payload = {
        "error_message": "Test error",
        "error_type": "Error",
    }

    response = await async_client.post("/log-error", json=payload)

    assert response.status_code == 403
    body = response.json()
    assert "detail" in body


@pytest.mark.asyncio
async def test_log_error_with_invalid_token(async_client):
    """Test error logging with invalid authentication token."""
    payload = {
        "error_message": "Test error",
        "error_type": "Error",
    }

    response = await async_client.post(
        "/log-error",
        json=payload,
        headers={"Authorization": "Bearer invalid_token"},
    )

    assert response.status_code == 401
    body = response.json()
    assert "detail" in body


@pytest.mark.asyncio
async def test_log_error_missing_required_fields(async_client):
    """Test error logging with missing required fields."""
    user, token = await create_test_user_and_token()
    try:
        payload = {
            "error_type": "Error",
        }

        response = await async_client.post(
            "/log-error",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 422

        payload = {
            "error_message": "Test error",
        }

        response = await async_client.post(
            "/log-error",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 422
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_log_error_default_component(async_client):
    """Test that component defaults to FRONTEND when not provided."""
    user, token = await create_test_user_and_token()
    try:
        payload = {
            "error_message": "Test error",
            "error_type": "Error",
        }

        response = await async_client.post(
            "/log-error",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()

        error_log = await ErrorLog.find_by_id(body["id"])
        assert error_log.component == "FRONTEND"

        await cleanup_models([error_log])
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_log_error_default_description(async_client):
    """Test that description defaults to error_message when not provided."""
    user, token = await create_test_user_and_token()
    try:
        payload = {
            "error_message": "Test error message",
            "error_type": "Error",
        }

        response = await async_client.post(
            "/log-error",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()

        error_log = await ErrorLog.find_by_id(body["id"])
        assert error_log.description == "Test error message"

        await cleanup_models([error_log])
    finally:
        await cleanup_models([user])

