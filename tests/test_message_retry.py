"""Regression tests for POST /conversations/{cid}/messages/{mid}/retry.

The agentic branch broke when ``generate_answer_agentic`` grew a seventh
return value (``artifact_ids``, artifact-storage PR #138): the retry handler
kept unpacking six and every agentic retry answered
500 "too many values to unpack (expected 6)".
"""

from unittest.mock import AsyncMock

import pytest

from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.schemas.generation_request import GenerationRequest
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token


async def _make_conversation_and_message(user, *, agentic: bool):
    conversation = Conversation(user_id=user.id, name="retry-test")
    await conversation.save()
    request_input = GenerationRequest(
        query="What is the bounding box of Rome?",
        llm_type="main",
        public_mcp_servers=["geocode"] if agentic else [],
    )
    message = await Message.create(
        conversation_id=conversation.id,
        input=request_input.query,
        output="",
        documents=[],
        use_rag=False,
        request_input=request_input,
        metadata={"error": "gateway timeout"},
    )
    return conversation, message


AGENTIC_SEVEN_TUPLE = (
    "retried answer",
    [],
    False,
    {"total_seconds": 1.0},
    {"system_prompt": "p"},
    [{"node": "agent"}],
    ["artifact-1"],
)


@pytest.mark.asyncio
async def test_retry_agentic_unpacks_seven_values(async_client, monkeypatch):
    user, token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _make_conversation_and_message(
            user, agentic=True
        )
        monkeypatch.setattr(
            "src.routers.message.generate_answer_agentic",
            AsyncMock(return_value=AGENTIC_SEVEN_TUPLE),
        )

        response = await async_client.post(
            f"/conversations/{conversation.id}/messages/{message.id}/retry",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["answer"] == "retried answer"
        assert body["conversation_id"] == conversation.id

        saved = await Message.find_by_id(message.id)
        assert saved.output == "retried answer"
        assert saved.artifact_ids == ["artifact-1"]
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


@pytest.mark.asyncio
async def test_retry_agentic_without_artifacts_stores_none(
    async_client, monkeypatch
):
    user, token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _make_conversation_and_message(
            user, agentic=True
        )
        no_artifacts = AGENTIC_SEVEN_TUPLE[:-1] + ([],)
        monkeypatch.setattr(
            "src.routers.message.generate_answer_agentic",
            AsyncMock(return_value=no_artifacts),
        )

        response = await async_client.post(
            f"/conversations/{conversation.id}/messages/{message.id}/retry",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200, response.text
        saved = await Message.find_by_id(message.id)
        assert saved.artifact_ids is None
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )


@pytest.mark.asyncio
async def test_retry_non_agentic_still_unpacks_six_values(
    async_client, monkeypatch
):
    user, token = await create_test_user_and_token()
    conversation = message = None
    try:
        conversation, message = await _make_conversation_and_message(
            user, agentic=False
        )
        monkeypatch.setattr(
            "src.routers.message.generate_answer",
            AsyncMock(
                return_value=(
                    "classic answer",
                    [],
                    False,
                    {"total_seconds": 1.0},
                    {"system_prompt": "p"},
                    {},
                )
            ),
        )

        response = await async_client.post(
            f"/conversations/{conversation.id}/messages/{message.id}/retry",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200, response.text
        assert response.json()["answer"] == "classic answer"
    finally:
        await cleanup_models(
            [doc for doc in (user, conversation, message) if doc is not None]
        )
