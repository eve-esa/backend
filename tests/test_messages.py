import pytest
import httpx
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from openai import AsyncOpenAI

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models
from src.database.models.message import Message
from src.core.llm_manager import LLMType
from server import app


# Mock the generate_answer for test speed reasons, remove the mock when actually want to test the answer generation from the LLM
async def mock_generate_answer(request, conversation_id=None, user_id=None):
    return "Test answer", [], False, {}, {}, []


async def mark_user_as_rate_limited(user):
    user.rate_limit_tokens_used = 1000
    user.rate_limit_period_start = datetime.now(timezone.utc) - timedelta(days=1)
    user.rate_limit_period_end = datetime.now(timezone.utc) + timedelta(days=1)
    await user.save()


def _auth_headers(token: str):
    return {"Authorization": f"Bearer {token}"}


def _patch_fake_llm(monkeypatch, *, invoke_text=None, stream_chunks=None):
    class _FakeLLM:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content=invoke_text or "")

        async def astream(self, _messages):
            for chunk in stream_chunks or []:
                yield SimpleNamespace(content=chunk)

    class _FakeLLMManager:
        def get_client_for_model(self, _model):
            return _FakeLLM()

    monkeypatch.setattr(
        "src.services.llm_inference.get_shared_llm_manager", lambda: _FakeLLMManager()
    )


@pytest.mark.asyncio
async def test_message_flow(async_client, monkeypatch):
    """Full flow: create conversation → add message → update feedback."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        create_conv_payload = {"name": "Msg Test Conversation"}
        conv_resp = await async_client.post(
            "/conversations",
            json=create_conv_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert conv_resp.status_code == 200
        conv_id = conv_resp.json()["id"]

        create_msg_payload = {
            "query": "Hello",
            "score_threshold": 0.6,
            "temperature": 0.0645,
            "k": 10,
            "filters": {
                "should": None, 
                "min_should": None,
                "must": [],
                "must_not": None
            },
            "llm_type": LLMType.Main.value,
            "public_collections": [ ]
        }
        msg_resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json=create_msg_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert msg_resp.status_code == 200
        msg_body = msg_resp.json()
        assert isinstance(msg_body["answer"], str) and msg_body["answer"]
        assert msg_body["conversation_id"] == conv_id
        msg_id = msg_body["id"]

        detail_resp = await async_client.get(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert detail_resp.status_code == 200
        assert len(detail_resp.json()["messages"]) == 1

        patch_payload = {"feedback": "positive", "was_copied": True}
        patch_resp = await async_client.patch(
            f"/conversations/{conv_id}/messages/{msg_id}",
            json=patch_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json() == {"message": "Feedback updated successfully"}

        updated = await Message.find_by_id(msg_id)
        assert updated is not None
        assert updated.feedback == patch_payload["feedback"]
        assert updated.was_copied is True

        del_resp = await async_client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert del_resp.status_code == 200

        # Message should be gone after conversation deletion
        assert await Message.find_by_id(msg_id) is None
    finally:
        await cleanup_models([user])


# ---------------- Negative cases -----------------


@pytest.mark.asyncio
async def test_create_message_nonexistent_conversation(async_client, monkeypatch):
    """Attempting to create a message for a non-existing conversation should return 404."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        fake_conv_id = "000000000000000000000000"
        payload = {"query": "Hello"}
        resp = await async_client.post(
            f"/conversations/{fake_conv_id}/messages",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Conversation not found"
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_create_message_not_owner(async_client, monkeypatch):
    """A user cannot add messages to someone else's conversation (403)."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    owner, owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        conv_resp = await async_client.post(
            "/conversations",
            json={"name": "Owner Conversation"},
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        conv_id = conv_resp.json()["id"]

        payload = {"query": "Hi"}
        resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json=payload,
            headers={"Authorization": f"Bearer {intr_token}"},
        )
        assert resp.status_code == 403
        assert (
            resp.json()["detail"]
            == "You are not allowed to add a message to this conversation"
        )

        # cleanup conversation
        await async_client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
    finally:
        await cleanup_models([owner, intruder])


@pytest.mark.asyncio
async def test_update_feedback_wrong_conversation(async_client, monkeypatch):
    """Updating feedback with mismatched conversation/message should yield 404."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        # First conversation with a message
        conv1_resp = await async_client.post(
            "/conversations",
            json={"name": "Conv1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        conv1_id = conv1_resp.json()["id"]

        msg_resp = await async_client.post(
            f"/conversations/{conv1_id}/messages",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        msg_id = msg_resp.json()["id"]

        # Second conversation
        conv2_id = (
            await async_client.post(
                "/conversations",
                json={"name": "Conv2"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        patch = {"feedback": "positive"}
        resp = await async_client.patch(
            f"/conversations/{conv2_id}/messages/{msg_id}",
            json=patch,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Message not found in this conversation"

        # cleanup
        await async_client.delete(
            f"/conversations/{conv1_id}", headers={"Authorization": f"Bearer {token}"}
        )
        await async_client.delete(
            f"/conversations/{conv2_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_update_feedback_not_owner(async_client, monkeypatch):
    """User cannot update feedback in conversation they don't own → 403."""

    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    owner, owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        conv_resp = await async_client.post(
            "/conversations",
            json={"name": "Owner Conv"},
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        conv_id = conv_resp.json()["id"]

        msg_id = (
            await async_client.post(
                f"/conversations/{conv_id}/messages",
                json={"query": "hi"},
                headers={"Authorization": f"Bearer {owner_token}"},
            )
        ).json()["id"]

        patch_payload = {"feedback": "negative"}
        resp = await async_client.patch(
            f"/conversations/{conv_id}/messages/{msg_id}",
            json=patch_payload,
            headers={"Authorization": f"Bearer {intr_token}"},
        )
        assert resp.status_code == 403
        assert (
            resp.json()["detail"]
            == "You are not allowed to update feedback for this message"
        )

        # cleanup
        await async_client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
    finally:
        await cleanup_models([owner, intruder])



@pytest.mark.asyncio
async def test_generate_llm_requires_auth(async_client):
    resp = await async_client.post(
        "/generate-llm",
        json={"query": "hello"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_generate_llm_rate_limited(async_client):
    user, token = await create_test_user_and_token()
    try:
        await mark_user_as_rate_limited(user)

        resp = await async_client.post(
            "/generate-llm",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 429
        assert "Token budget exceeded" in resp.json()["detail"]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_generate_llm_consumes_tokens(async_client, monkeypatch):
    class _FakeLLM:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content="mocked answer")

    class _FakeLLMManager:
        def get_client_for_model(self, _model):
            return _FakeLLM()

    monkeypatch.setattr(
        "src.services.llm_inference.get_shared_llm_manager", lambda: _FakeLLMManager()
    )

    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.post(
            "/generate-llm",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

        refreshed = await type(user).find_by_id(user.id)
        assert refreshed is not None
        assert refreshed.rate_limit_tokens_used > 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_openai_chat_completions_requires_auth(async_client):
    resp = await async_client.post(
        "/v1/chat/completions",
        json={
            "model": LLMType.Main.value,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_openai_chat_completions_consumes_tokens(async_client, monkeypatch):
    _patch_fake_llm(monkeypatch, invoke_text="mocked openai answer")

    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.post(
            "/v1/chat/completions",
            json={
                "model": LLMType.Main.value,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == LLMType.Main.value
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "mocked openai answer"
        assert body["usage"]["total_tokens"] > 0

        refreshed = await type(user).find_by_id(user.id)
        assert refreshed is not None
        assert refreshed.rate_limit_tokens_used > 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_openai_chat_completions_stream_consumes_tokens(async_client, monkeypatch):
    _patch_fake_llm(monkeypatch, stream_chunks=["mocked ", "stream"])

    user, token = await create_test_user_and_token()
    try:
        async with async_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": LLMType.Main.value,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=_auth_headers(token),
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = await resp.aread()

        text = body.decode()
        assert '"object": "chat.completion.chunk"' in text
        assert '"role": "assistant"' in text
        assert '"content": "mocked "' in text
        assert '"content": "stream"' in text
        assert '"finish_reason": "stop"' in text
        assert "data: [DONE]" in text

        refreshed = await type(user).find_by_id(user.id)
        assert refreshed is not None
        assert refreshed.rate_limit_tokens_used > 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_openai_completions_requires_auth(async_client):
    resp = await async_client.post(
        "/v1/completions",
        json={"model": LLMType.Main.value, "prompt": "hello"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_openai_completions_consumes_tokens(async_client, monkeypatch):
    _patch_fake_llm(monkeypatch, invoke_text="mocked completion answer")

    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.post(
            "/v1/completions",
            json={"model": LLMType.Main.value, "prompt": "hello"},
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["model"] == LLMType.Main.value
        assert body["choices"][0]["text"] == "mocked completion answer"
        assert body["usage"]["total_tokens"] > 0

        refreshed = await type(user).find_by_id(user.id)
        assert refreshed is not None
        assert refreshed.rate_limit_tokens_used > 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_openai_completions_stream_consumes_tokens(async_client, monkeypatch):
    _patch_fake_llm(monkeypatch, stream_chunks=["mocked ", "completion"])

    user, token = await create_test_user_and_token()
    try:
        async with async_client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": LLMType.Main.value,
                "stream": True,
                "prompt": "hello",
            },
            headers=_auth_headers(token),
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = await resp.aread()

        text = body.decode()
        assert '"object": "text_completion"' in text
        assert '"text": "mocked "' in text
        assert '"text": "completion"' in text
        assert '"finish_reason": "stop"' in text
        assert "data: [DONE]" in text

        refreshed = await type(user).find_by_id(user.id)
        assert refreshed is not None
        assert refreshed.rate_limit_tokens_used > 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_openai_models_lists_supported_models(async_client):
    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.get(
            "/v1/models",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        model_ids = {model["id"] for model in body["data"]}
        assert LLMType.Main.value in model_ids
        assert LLMType.Fallback.value in model_ids
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_openai_python_client_chat_completions(monkeypatch):
    _patch_fake_llm(monkeypatch, invoke_text="sdk answer")

    user, token = await create_test_user_and_token()
    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key=token,
        http_client=http_client,
    )
    try:
        resp = await client.chat.completions.create(
            model=LLMType.Main.value,
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert resp.choices[0].message.content == "sdk answer"
    finally:
        await client.close()
        await http_client.aclose()
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_generate_rate_limited(async_client):
    user, token = await create_test_user_and_token()
    try:
        await mark_user_as_rate_limited(user)

        resp = await async_client.post(
            "/generate",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 429
        assert "Token budget exceeded" in resp.json()["detail"]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_retrieve_rate_limited(async_client):
    user, token = await create_test_user_and_token()
    try:
        await mark_user_as_rate_limited(user)

        resp = await async_client.post(
            "/retrieve",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 429
        assert "Token budget exceeded" in resp.json()["detail"]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_retrieve_consumes_tokens(async_client, monkeypatch):
    class _RagDecision:
        requery = "rewritten query"

    async def _mock_should_use_rag(*_args, **_kwargs):
        return _RagDecision(), None, None

    async def _mock_setup_rag_and_context(_request):
        return "", [], {"retrieve": 0.01}, [{"chunk": "example"}]

    monkeypatch.setattr("src.routers.message.should_use_rag", _mock_should_use_rag)
    monkeypatch.setattr(
        "src.routers.message.setup_rag_and_context", _mock_setup_rag_and_context
    )

    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.post(
            "/retrieve",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

        refreshed = await type(user).find_by_id(user.id)
        assert refreshed is not None
        assert refreshed.rate_limit_tokens_used > 0
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_stream_messages_rate_limited(async_client):
    user, token = await create_test_user_and_token()
    try:
        conv_resp = await async_client.post(
            "/conversations",
            json={"name": "Stream Rate Limit Conversation"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert conv_resp.status_code == 200
        conv_id = conv_resp.json()["id"]

        await mark_user_as_rate_limited(user)

        resp = await async_client.post(
            f"/conversations/{conv_id}/stream_messages",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 429
        assert "Token budget exceeded" in resp.json()["detail"]
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_stream_hallucination_rate_limited(async_client, monkeypatch):
    monkeypatch.setattr("src.routers.message.generate_answer", mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        conv_resp = await async_client.post(
            "/conversations",
            json={"name": "Hallucination Stream Rate Limit Conversation"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert conv_resp.status_code == 200
        conv_id = conv_resp.json()["id"]

        msg_resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json={"query": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert msg_resp.status_code == 200
        msg_id = msg_resp.json()["id"]

        await mark_user_as_rate_limited(user)

        resp = await async_client.post(
            f"/conversations/{conv_id}/messages/{msg_id}/stream-hallucination",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 429
        assert "Token budget exceeded" in resp.json()["detail"]
    finally:
        await cleanup_models([user])