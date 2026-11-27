"""Owner delete of an artifact and the conversation delete cascade.

A message references a file as an MCP output (``artifact_ids``) or as an
attached upload (``attachments[].image_id``); both are covered.
"""

import asyncio
import io
import logging
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
from botocore.exceptions import ClientError

from src.database.models.artifact import Artifact, ArtifactSource
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.database.mongo import get_collection
from src.routers import message as message_router
from src.services import artifact_cleanup
from src.services.agents.core.tool_results import tool_result_prefix
from src.services.cancel_manager import CancelManager
from src.services.storage import StorageService
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
LOGGER = "src.services.artifact_cleanup"


class _FailingDeleteStorage(FakeStorage):
    """Every delete raises the error S3 gives a role without the permission."""

    def _denied(self, op):
        return ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "secret-key-material"}}, op
        )

    async def delete_object(self, key):
        raise self._denied("DeleteObject")

    async def delete_prefix(self, prefix, max_keys=artifact_cleanup.PREFIX_DELETE_MAX_KEYS):
        raise self._denied("DeleteObjects")


def _use(monkeypatch, fake):
    monkeypatch.setattr("src.routers.artifact.storage_service", fake)
    monkeypatch.setattr("src.services.artifact_move.storage_service", fake)
    monkeypatch.setattr("src.services.artifact_cleanup.storage_service", fake)
    return fake


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


async def _seed(fake, user_id, conversation_id=None, filename="pic.png"):
    key = fake.build_user_key(user_id, "png", prefix="artifacts")
    await fake.put_object(key, PNG_BYTES, "image/png")
    return await Artifact.create(
        user_id=user_id,
        key=key,
        filename=filename,
        content_type="image/png",
        size_bytes=len(PNG_BYTES),
        source=ArtifactSource(type="upload"),
        conversation_id=conversation_id,
    )


async def _mock_generate_answer(request, conversation_id=None, user_id=None):
    return "Test answer", [], False, {}, {}, []


async def _upload(async_client, token):
    resp = await async_client.post(
        "/artifacts",
        headers=_auth(token),
        files={"file": ("pic.png", io.BytesIO(PNG_BYTES), "image/png")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


async def _attach(async_client, token, conversation_id, artifact_id):
    """Attach an upload the way the chat does: a message with ``artifact_ids``."""
    resp = await async_client.post(
        f"/conversations/{conversation_id}/messages",
        json={"query": "look", "artifact_ids": [artifact_id]},
        headers=_auth(token),
    )
    assert resp.status_code == 200, resp.text


# ---------------- DELETE /artifacts/{id} -----------------


async def test_owner_delete_removes_row_object_and_message_refs(async_client, monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    artifact = await _seed(fake, user.id, conversation_id="c1")
    message = await Message.create(
        conversation_id="c1", input="q", output="a", artifact_ids=[artifact.id, "other"]
    )
    try:
        resp = await async_client.delete(f"/artifacts/{artifact.id}", headers=_auth(token))

        assert resp.status_code == 200
        assert await Artifact.find_by_id(artifact.id) is None
        assert artifact.key not in fake.objects
        assert (await Message.find_by_id(message.id)).artifact_ids == ["other"]
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, message])


async def test_owner_delete_pulls_an_attached_upload(async_client, monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    monkeypatch.setattr("src.routers.message.generate_answer", _mock_generate_answer)
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Attach")
    try:
        artifact_id = await _upload(async_client, token)
        await _attach(async_client, token, conversation.id, artifact_id)
        sibling = await Message.create(
            conversation_id=conversation.id,
            input="q",
            output="a",
            attachments=[{"image_id": "keep-me"}, {"image_id": artifact_id}],
        )
        # A message with no attachments (null) must not break the pull.
        await Message.create(
            conversation_id=conversation.id, input="q", output="a", artifact_ids=[artifact_id]
        )

        resp = await async_client.delete(f"/artifacts/{artifact_id}", headers=_auth(token))

        assert resp.status_code == 200
        assert not fake.objects
        left = await Message.get_collection().count_documents(
            {
                "$or": [
                    {"artifact_ids": artifact_id},
                    {"attachments.image_id": artifact_id},
                ]
            }
        )
        assert left == 0
        assert (await Message.find_by_id(sibling.id)).attachments == [{"image_id": "keep-me"}]
    finally:
        await Message.delete_many({"conversation_id": conversation.id})
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_delete_of_a_foreign_artifact_is_refused_and_keeps_it(async_client, monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    owner, _ = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    artifact = await _seed(fake, owner.id)
    try:
        resp = await async_client.delete(f"/artifacts/{artifact.id}", headers=_auth(intr_token))

        # 404, like a missing id: the answer never confirms someone else's file.
        assert resp.status_code == 404
        assert await Artifact.find_by_id(artifact.id) is not None
        assert artifact.key in fake.objects
    finally:
        await Artifact.delete_many({"user_id": owner.id})
        await cleanup_models([owner, intruder])


async def test_owner_delete_reads_the_row_on_the_primary(async_client, monkeypatch):
    """A delete right after an upload must not miss the row on a lagging secondary."""
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    artifact = await _seed(fake, user.id)
    primary = Artifact.get_primary_collection()

    class _LaggingSecondary:
        def find_one(self, *args, **kwargs):
            return primary.find_one({"_id": None})

        def __getattr__(self, name):
            return getattr(primary, name)

    monkeypatch.setattr(Artifact, "get_collection", classmethod(lambda cls: _LaggingSecondary()))
    monkeypatch.setattr(Artifact, "get_primary_collection", classmethod(lambda cls: primary))
    try:
        resp = await async_client.delete(f"/artifacts/{artifact.id}", headers=_auth(token))
        assert resp.status_code == 200, resp.text
        assert artifact.key not in fake.objects
    finally:
        monkeypatch.undo()
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


async def test_delete_of_a_missing_artifact_is_404(async_client, monkeypatch):
    _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    try:
        for bad in ("0123456789abcdef01234567", "not-an-object-id"):
            resp = await async_client.delete(f"/artifacts/{bad}", headers=_auth(token))
            assert resp.status_code == 404
    finally:
        await cleanup_models([user])


async def test_storage_error_keeps_the_row_and_answers_503(async_client, monkeypatch, caplog):
    fake = _use(monkeypatch, _FailingDeleteStorage())
    user, token = await create_test_user_and_token()
    artifact = await _seed(fake, user.id)
    try:
        with caplog.at_level(logging.WARNING):
            resp = await async_client.delete(f"/artifacts/{artifact.id}", headers=_auth(token))

        assert resp.status_code == 503
        assert resp.json()["detail"]["code"] == "artifact_delete_failed"
        assert await Artifact.find_by_id(artifact.id) is not None
        assert "secret-key-material" not in caplog.text
        assert artifact.id in caplog.text
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


async def test_a_failed_pull_after_the_row_is_gone_still_answers_200(
    async_client, monkeypatch, caplog
):
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    artifact = await _seed(fake, user.id)

    async def _broken_pull(artifact_id):
        raise RuntimeError("mongo hiccup")

    monkeypatch.setattr(artifact_cleanup, "pull_message_references", _broken_pull)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            resp = await async_client.delete(f"/artifacts/{artifact.id}", headers=_auth(token))

        assert resp.status_code == 200
        assert await Artifact.find_by_id(artifact.id) is None
        assert "artifact.pull_failed" in caplog.text
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


async def test_quota_counter_untouched_by_delete(async_client, monkeypatch):
    """Deleting does not hand the daily upload slot back (the cap is a rate)."""
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    try:
        artifact_id = await _upload(async_client, token)
        before = await get_collection("image_upload_quota").find_one({"user_id": user.id})
        await async_client.delete(f"/artifacts/{artifact_id}", headers=_auth(token))
        after = await get_collection("image_upload_quota").find_one({"user_id": user.id})
        assert before["count"] == after["count"] == 1
        assert not fake.objects
    finally:
        await get_collection("image_upload_quota").delete_many({"user_id": user.id})
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- Conversation delete cascade -----------------


async def _conversation_with_files(fake, user_id):
    conversation = await Conversation.create(user_id=user_id, name="Files")
    artifact = await _seed(fake, user_id, conversation_id=conversation.id)
    prefix = tool_result_prefix(user_id, conversation.id)
    await fake.put_object(f"{prefix}call_1", b"big tool output", "text/plain")
    await fake.put_object(f"{prefix}call_2", b"more output", "text/plain")
    return conversation, artifact, prefix


async def test_conversation_delete_removes_artifacts_and_tool_results(async_client, monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation, artifact, prefix = await _conversation_with_files(fake, user.id)
    other = await _seed(fake, user.id, conversation_id="unrelated")
    try:
        resp = await async_client.delete(f"/conversations/{conversation.id}", headers=_auth(token))

        assert resp.status_code == 200
        assert await Artifact.find_by_id(artifact.id) is None
        assert artifact.key not in fake.objects
        assert not [k for k in fake.objects if k.startswith(prefix)]
        assert await Artifact.find_by_id(other.id) is not None
        assert other.key in fake.objects
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_conversation_delete_with_zero_artifacts_answers_200(async_client, monkeypatch):
    _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Empty")
    try:
        resp = await async_client.delete(f"/conversations/{conversation.id}", headers=_auth(token))
        assert resp.status_code == 200
        assert await Conversation.find_by_id(conversation.id) is None
    finally:
        await cleanup_models([user, conversation])


async def test_mcp_output_shown_in_another_conversation_survives(async_client, monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation, artifact, _ = await _conversation_with_files(fake, user.id)
    second = await Conversation.create(user_id=user.id, name="Second")
    reuse = await Message.create(
        conversation_id=second.id, input="q", output="a", artifact_ids=[artifact.id]
    )
    try:
        resp = await async_client.delete(f"/conversations/{conversation.id}", headers=_auth(token))

        assert resp.status_code == 200
        kept = await Artifact.find_by_id(artifact.id)
        assert kept is not None
        assert kept.conversation_id == second.id
        # The object moved with the row. The sweep would have deleted it in
        # the conversation that just went away.
        assert kept.key in fake.objects
        assert artifact.key not in fake.objects
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation, second, reuse])


async def test_a_foreign_users_message_does_not_keep_the_file(async_client, monkeypatch):
    """Only the owner's conversations count as showing the file elsewhere."""
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    stranger, _ = await create_test_user_and_token()
    conversation, artifact, _ = await _conversation_with_files(fake, user.id)
    foreign = await Conversation.create(user_id=stranger.id, name="Not yours")
    copied = await Message.create(
        conversation_id=foreign.id, input="q", output="a", artifact_ids=[artifact.id]
    )
    try:
        resp = await async_client.delete(f"/conversations/{conversation.id}", headers=_auth(token))

        assert resp.status_code == 200
        assert await Artifact.find_by_id(artifact.id) is None
        assert artifact.key not in fake.objects
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, stranger, conversation, foreign, copied])


async def test_upload_attached_in_two_conversations_survives_the_delete_of_one(
    async_client, monkeypatch
):
    """Attached in A, then in B (the row moves to B); deleting B keeps it for A."""
    fake = _use(monkeypatch, FakeStorage())
    monkeypatch.setattr("src.routers.message.generate_answer", _mock_generate_answer)
    user, token = await create_test_user_and_token()
    first = await Conversation.create(user_id=user.id, name="A")
    second = await Conversation.create(user_id=user.id, name="B")
    try:
        artifact_id = await _upload(async_client, token)
        await _attach(async_client, token, first.id, artifact_id)
        await _attach(async_client, token, second.id, artifact_id)
        assert (await Artifact.find_by_id(artifact_id)).conversation_id == second.id
        key = (await Artifact.find_by_id(artifact_id)).key

        resp = await async_client.delete(f"/conversations/{second.id}", headers=_auth(token))

        assert resp.status_code == 200
        kept = await Artifact.find_by_id(artifact_id)
        assert kept is not None and kept.conversation_id == first.id
        assert kept.key in fake.objects
        assert key not in fake.objects
        moved = kept.key
        got = await async_client.get(f"/artifacts/{artifact_id}", headers=_auth(token))
        assert got.status_code == 200

        # Deleting the last conversation that shows it erases it.
        resp = await async_client.delete(f"/conversations/{first.id}", headers=_auth(token))
        assert resp.status_code == 200
        assert await Artifact.find_by_id(artifact_id) is None
        assert moved not in fake.objects
    finally:
        await Message.delete_many({"conversation_id": {"$in": [first.id, second.id]}})
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, first, second])


async def test_shared_file_check_reads_the_primary(monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    user, _ = await create_test_user_and_token()
    conversation, artifact, _ = await _conversation_with_files(fake, user.id)
    second = await Conversation.create(user_id=user.id, name="Second")
    reuse = await Message.create(
        conversation_id=second.id,
        input="q",
        output="a",
        attachments=[{"image_id": artifact.id}],
    )

    primary = Message.get_primary_collection()

    class _LaggingSecondary:
        """A secondary that has not seen the attach: every find is empty."""

        def find(self, *args, **kwargs):
            return primary.find({"_id": None})

        def __getattr__(self, name):
            return getattr(primary, name)

    monkeypatch.setattr(Message, "get_collection", classmethod(lambda cls: _LaggingSecondary()))
    monkeypatch.setattr(Message, "get_primary_collection", classmethod(lambda cls: primary))
    try:
        found = await artifact_cleanup._other_conversations(
            user.id, [artifact.id], conversation.id
        )
        assert found == {artifact.id: second.id}
    finally:
        monkeypatch.undo()
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation, second, reuse])


async def _messages(conversation_id):
    return await Message.count_documents({"conversation_id": conversation_id})


def _assert_incomplete(resp):
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"]["code"] == "conversation_delete_incomplete"


async def test_s3_error_on_an_artifact_keeps_the_conversation_and_answers_503(
    async_client, monkeypatch, caplog
):
    """The object delete fails; tool results are still removed; a retry can finish."""

    class _ObjectDeleteDenied(FakeStorage):
        async def delete_object(self, key):
            raise _FailingDeleteStorage()._denied("DeleteObject")

    fake = _use(monkeypatch, _ObjectDeleteDenied())
    user, token = await create_test_user_and_token()
    conversation, artifact, prefix = await _conversation_with_files(fake, user.id)
    await Message.create(conversation_id=conversation.id, input="q", output="a")
    try:
        with caplog.at_level(logging.WARNING):
            resp = await async_client.delete(
                f"/conversations/{conversation.id}", headers=_auth(token)
            )

        _assert_incomplete(resp)
        assert await Conversation.find_by_id(conversation.id) is not None
        assert await _messages(conversation.id) == 1
        # The row stays with its object, so the retry still finds both.
        assert await Artifact.find_by_id(artifact.id) is not None
        assert artifact.key in fake.objects
        assert not [k for k in fake.objects if k.startswith(prefix)]
        records = [r for r in caplog.records if r.name == LOGGER]
        assert records and all(r.levelno == logging.WARNING for r in records)
        assert conversation.id in caplog.text and artifact.id in caplog.text
        assert "secret-key-material" not in caplog.text
        assert "pic.png" not in caplog.text
    finally:
        await Message.delete_many({"conversation_id": conversation.id})
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_s3_error_on_tool_results_keeps_the_conversation_and_answers_503(
    async_client, monkeypatch
):
    class _PrefixDeleteDenied(FakeStorage):
        async def delete_prefix(self, prefix, max_keys=artifact_cleanup.PREFIX_DELETE_MAX_KEYS):
            raise _FailingDeleteStorage()._denied("DeleteObjects")

    fake = _use(monkeypatch, _PrefixDeleteDenied())
    user, token = await create_test_user_and_token()
    conversation, artifact, prefix = await _conversation_with_files(fake, user.id)
    try:
        resp = await async_client.delete(f"/conversations/{conversation.id}", headers=_auth(token))

        _assert_incomplete(resp)
        assert await Conversation.find_by_id(conversation.id) is not None
        # Independent: the artifact went even though the tool results did not.
        assert await Artifact.find_by_id(artifact.id) is None
        assert [k for k in fake.objects if k.startswith(prefix)]
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_mongo_error_in_the_cascade_keeps_the_conversation_and_still_removes_tool_results(
    async_client, monkeypatch, caplog
):
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation, artifact, prefix = await _conversation_with_files(fake, user.id)
    await Message.create(conversation_id=conversation.id, input="q", output="a")

    async def _mongo_down(*args, **kwargs):
        raise RuntimeError("connection reset with user text inside")

    monkeypatch.setattr(artifact_cleanup, "_other_conversations", _mongo_down)
    try:
        with caplog.at_level(logging.WARNING):
            resp = await async_client.delete(
                f"/conversations/{conversation.id}", headers=_auth(token)
            )

        _assert_incomplete(resp)
        assert await Conversation.find_by_id(conversation.id) is not None
        assert await _messages(conversation.id) == 1
        assert await Artifact.find_by_id(artifact.id) is not None
        assert not [k for k in fake.objects if k.startswith(prefix)]
        assert "user text inside" not in caplog.text
        assert "user text inside" not in resp.text
    finally:
        await Message.delete_many({"conversation_id": conversation.id})
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_deadline_on_the_artifacts_still_removes_tool_results_and_answers_503(
    async_client, monkeypatch, caplog
):
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation, artifact, prefix = await _conversation_with_files(fake, user.id)

    async def _hangs(*args, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(artifact_cleanup, "_other_conversations", _hangs)
    monkeypatch.setattr(artifact_cleanup, "CASCADE_TIMEOUT_S", 0.2)
    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            resp = await async_client.delete(
                f"/conversations/{conversation.id}", headers=_auth(token)
            )

        assert time.monotonic() - started < 5
        _assert_incomplete(resp)
        assert "reason=timeout" in caplog.text
        assert await Conversation.find_by_id(conversation.id) is not None
        assert await Artifact.find_by_id(artifact.id) is not None
        assert not [k for k in fake.objects if k.startswith(prefix)]
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_a_retry_after_a_failed_cascade_finishes_the_job(async_client, monkeypatch):
    user, token = await create_test_user_and_token()
    fake = FakeStorage()
    conversation, artifact, prefix = await _conversation_with_files(fake, user.id)
    _use(monkeypatch, _FailingDeleteStorage())
    try:
        first = await async_client.delete(f"/conversations/{conversation.id}", headers=_auth(token))
        _assert_incomplete(first)

        _use(monkeypatch, fake)
        second = await async_client.delete(
            f"/conversations/{conversation.id}", headers=_auth(token)
        )

        assert second.status_code == 200
        assert await Conversation.find_by_id(conversation.id) is None
        assert await Artifact.find_by_id(artifact.id) is None
        assert not fake.objects
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_the_cascade_pages_past_one_batch(monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    monkeypatch.setattr(artifact_cleanup, "CASCADE_BATCH_SIZE", 2)
    user, _ = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Many")
    for i in range(5):
        await _seed(fake, user.id, conversation_id=conversation.id, filename=f"f{i}.png")
    try:
        await artifact_cleanup.delete_conversation_artifacts(user.id, conversation.id)

        assert await Artifact.count_documents({"conversation_id": conversation.id}) == 0
        assert not fake.objects
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_a_capped_tool_result_delete_reports_itself_incomplete(monkeypatch):
    """Tool results have no row: a run that hit the key cap must ask for a retry."""
    fake = _use(monkeypatch, FakeStorage())
    monkeypatch.setattr(artifact_cleanup, "PREFIX_DELETE_MAX_KEYS", 1)
    user, _ = await create_test_user_and_token()
    conversation, _, prefix = await _conversation_with_files(fake, user.id)
    try:
        with pytest.raises(artifact_cleanup.ArtifactCascadeIncomplete) as err:
            await artifact_cleanup.delete_conversation_artifacts(user.id, conversation.id)
        assert err.value.failed == ["tool_results"]
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_cascade_is_idempotent(monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    user, _ = await create_test_user_and_token()
    conversation, artifact, prefix = await _conversation_with_files(fake, user.id)
    try:
        await artifact_cleanup.delete_conversation_artifacts(user.id, conversation.id)
        await artifact_cleanup.delete_conversation_artifacts(user.id, conversation.id)

        assert await Artifact.find_by_id(artifact.id) is None
        assert not [k for k in fake.objects if k.startswith(prefix)]
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_a_hung_storage_thread_is_bounded(monkeypatch, caplog):
    """The real ``delete_prefix`` runs a threadpool call that cannot be cancelled."""
    release = threading.Event()
    service = StorageService()

    def _blocking_delete_prefix_sync(prefix, max_keys=0):
        release.wait(timeout=10)
        return 0, 0

    monkeypatch.setattr(service, "delete_prefix_sync", _blocking_delete_prefix_sync)
    monkeypatch.setattr("src.services.artifact_cleanup.storage_service", service)
    monkeypatch.setattr(artifact_cleanup, "CASCADE_TIMEOUT_S", 0.2)
    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(artifact_cleanup.ArtifactCascadeIncomplete):
                await artifact_cleanup.delete_conversation_artifacts("u1", "c1")
        assert time.monotonic() - started < 2
        assert "reason=timeout" in caplog.text
    finally:
        release.set()


# ---------------- A running turn is stopped before the cascade -----------------


async def test_conversation_delete_stops_the_running_turn_before_erasing(
    async_client, monkeypatch
):
    """The turn persists its partial answer as it unwinds; that write lands
    before the delete, so nothing of the conversation is left behind."""
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation, artifact, _ = await _conversation_with_files(fake, user.id)
    mgr = CancelManager()
    monkeypatch.setattr(mgr, "_is_redis_enabled", lambda: False)
    unwound = asyncio.Event()

    async def _turn():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            await Message.create(conversation_id=conversation.id, input="q", output="partial")
            unwound.set()
            raise
        finally:
            mgr.clear_mapping_for(conversation.id, "m1")
            mgr.clear("m1")

    mgr.create("m1")
    mgr.link_conversation(conversation.id, "m1")
    task = asyncio.create_task(_turn())
    mgr.set_task("m1", task)
    await asyncio.sleep(0)
    bus = AsyncMock()
    try:
        with patch.object(message_router, "get_cancel_manager", return_value=mgr), patch.object(
            message_router, "get_stream_bus", return_value=bus
        ):
            resp = await async_client.delete(
                f"/conversations/{conversation.id}", headers=_auth(token)
            )

        assert resp.status_code == 200
        assert task.cancelled() and unwound.is_set()
        bus.stop.assert_awaited_once()
        assert await Message.count_documents({"conversation_id": conversation.id}) == 0
        assert await Artifact.find_by_id(artifact.id) is None
    finally:
        await Message.delete_many({"conversation_id": conversation.id})
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user, conversation])


async def test_conversation_delete_goes_on_when_the_turn_does_not_unwind(
    async_client, monkeypatch, caplog
):
    _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Stuck")

    async def _never_unwinds(conversation_id, timeout_s):
        return False

    monkeypatch.setattr(
        "src.routers.conversation.stop_active_generation", _never_unwinds
    )
    try:
        with caplog.at_level(logging.WARNING):
            resp = await async_client.delete(
                f"/conversations/{conversation.id}", headers=_auth(token)
            )
        assert resp.status_code == 200
        assert "conversation.delete.preempt_timeout" in caplog.text
        assert await Conversation.find_by_id(conversation.id) is None
    finally:
        await cleanup_models([user, conversation])


# ---------------- Indexes and the S3 client -----------------


async def test_reference_lookups_are_indexed():
    indexes = await Message.get_collection().index_information()
    assert indexes["messages_artifact_ids"]["key"] == [("artifact_ids", 1)]
    assert indexes["messages_attachments_image_id"]["key"] == [("attachments.image_id", 1)]


def test_s3_client_has_connect_read_and_retry_bounds():
    from src.config import S3_CONNECT_TIMEOUT_S, S3_MAX_ATTEMPTS, S3_READ_TIMEOUT_S

    config = StorageService()._client().meta.config
    assert config.connect_timeout == S3_CONNECT_TIMEOUT_S
    assert config.read_timeout == S3_READ_TIMEOUT_S
    assert config.retries["total_max_attempts"] == S3_MAX_ATTEMPTS
    assert config.retries["mode"] == "standard"


# ---------------- StorageService.delete_prefix_sync -----------------


class _StubS3:
    def __init__(self, keys, errors=None):
        self.keys = list(keys)
        self.errors = errors or []
        self.deleted = []
        self.list_calls = []

    def list_objects_v2(self, **params):
        self.list_calls.append(params)
        matching = [k for k in self.keys if k.startswith(params["Prefix"])]
        page = matching[: params["MaxKeys"]]
        return {"Contents": [{"Key": k} for k in page], "IsTruncated": len(page) < len(matching)}

    def delete_objects(self, Bucket, Delete):
        gone = [o["Key"] for o in Delete["Objects"]]
        self.deleted.extend(gone)
        if not self.errors:
            self.keys = [k for k in self.keys if k not in set(gone)]
        return {"Errors": self.errors}


def _service_with(monkeypatch, stub):
    service = StorageService()
    monkeypatch.setattr(service, "_client", lambda: stub)
    return service


def test_delete_prefix_pages_and_batches(monkeypatch):
    keys = [f"users/u1/conversations/c1/large_tool_results/k{i}" for i in range(2500)]
    stub = _StubS3(keys + ["users/u1/artifacts/keep.png"])
    service = _service_with(monkeypatch, stub)

    deleted, total_bytes = service.delete_prefix_sync(
        "users/u1/conversations/c1/large_tool_results/"
    )

    assert deleted == 2500
    assert total_bytes == 0
    assert sorted(stub.deleted) == sorted(keys)
    assert stub.keys == ["users/u1/artifacts/keep.png"]


@pytest.mark.parametrize("prefix", ["", "/", "users/", "users/u1", "migration/", "users//x/"])
def test_delete_prefix_refuses_a_broad_prefix(monkeypatch, prefix):
    stub = _StubS3(["users/u1/x"])
    service = _service_with(monkeypatch, stub)
    with pytest.raises(ValueError):
        service.delete_prefix_sync(prefix)
    assert stub.list_calls == [] and stub.deleted == []


def test_delete_prefix_raises_on_per_key_errors(monkeypatch):
    stub = _StubS3(
        ["users/u1/conversations/c1/large_tool_results/a"],
        errors=[{"Key": "a", "Code": "AccessDenied"}],
    )
    service = _service_with(monkeypatch, stub)
    with pytest.raises(RuntimeError):
        service.delete_prefix_sync("users/u1/conversations/c1/large_tool_results/")
