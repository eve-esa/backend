"""An artifact row never points to a missing object.

Found on dev: a composer image attached by two overlapping requests left its
row on the upload key after the other request had moved the object away, so
the thumbnail answered 404. The move and the cascade wrote a stale copy of
the row back over the fresh one (``save()`` is a full replace).
"""

import asyncio
import io
import logging

import pytest

from src.database.models.artifact import Artifact
from src.database.models.conversation import Conversation
from src.routers.artifact import move_to_conversation
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _use_fake_storage(monkeypatch) -> FakeStorage:
    fake = FakeStorage()
    monkeypatch.setattr("src.routers.artifact.storage_service", fake)
    monkeypatch.setattr("src.services.artifact_cleanup.storage_service", fake)
    return fake


async def _upload(async_client, token, conversation_id=None):
    data = {"conversation_id": conversation_id} if conversation_id else None
    return await async_client.post(
        "/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("pic.png", io.BytesIO(PNG_BYTES), "image/png")},
        data=data,
    )


async def _conversation(user_id: str, name: str) -> str:
    return (await Conversation.create(user_id=user_id, name=name)).id


async def _cleanup(user):
    from src.database.mongo import get_collection

    await Artifact.delete_many({"user_id": user.id})
    await Conversation.delete_many({"user_id": user.id})
    await get_collection("image_upload_quota").delete_many({"user_id": user.id})
    await cleanup_models([user])


async def _served(async_client, token, artifact_id):
    return await async_client.get(
        f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
    )


@pytest.mark.asyncio
async def test_overlapping_attach_keeps_the_row_on_a_live_object(async_client, monkeypatch):
    """Two requests read the row before either moves it: the second must not
    write the deleted upload key back."""
    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        image_id = (await _upload(async_client, token)).json()["id"]
        conv_a = await _conversation(user.id, "A")
        conv_b = await _conversation(user.id, "B")
        first = await Artifact.find_by_id_on_primary(image_id)
        second = await Artifact.find_by_id_on_primary(image_id)

        await move_to_conversation(first, conv_a)
        await move_to_conversation(second, conv_b)

        row = await Artifact.find_by_id_on_primary(image_id)
        assert row.key in fake.objects
        assert row.conversation_id == conv_b
        assert second.key == row.key
        assert (await _served(async_client, token, image_id)).status_code == 200
        # The loser's first copy does not linger next to the live object.
        assert list(fake.objects) == [row.key]
    finally:
        await _cleanup(user)


@pytest.mark.asyncio
async def test_concurrent_attach_to_the_same_conversation(async_client, monkeypatch):
    """Both requests copy to the same key: the object survives the loser."""
    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        image_id = (await _upload(async_client, token)).json()["id"]
        conv = await _conversation(user.id, "Same")
        first = await Artifact.find_by_id_on_primary(image_id)
        second = await Artifact.find_by_id_on_primary(image_id)

        await asyncio.gather(
            move_to_conversation(first, conv), move_to_conversation(second, conv)
        )

        row = await Artifact.find_by_id_on_primary(image_id)
        assert row.conversation_id == conv
        assert row.key in fake.objects
        assert list(fake.objects) == [row.key]
        assert (await _served(async_client, token, image_id)).status_code == 200
    finally:
        await _cleanup(user)


@pytest.mark.asyncio
async def test_cascade_move_does_not_restore_a_stale_key(async_client, monkeypatch):
    """A conversation delete that keeps an artifact for another conversation
    must not overwrite a key an attach changed meanwhile."""
    fake = _use_fake_storage(monkeypatch)
    from src.services import artifact_cleanup

    user, token = await create_test_user_and_token()
    try:
        doomed = await _conversation(user.id, "Doomed")
        kept = await _conversation(user.id, "Kept")
        image_id = (await _upload(async_client, token, doomed)).json()["id"]

        real_lookup = artifact_cleanup._other_conversations

        async def lookup_then_attach(user_id, artifact_ids, conversation_id):
            # The cascade has read the row; an attach to "kept" moves it now.
            fresh = await Artifact.find_by_id_on_primary(image_id)
            await move_to_conversation(fresh, kept)
            return {image_id: kept}

        monkeypatch.setattr(artifact_cleanup, "_other_conversations", lookup_then_attach)
        await artifact_cleanup.delete_conversation_artifacts(user.id, doomed)
        monkeypatch.setattr(artifact_cleanup, "_other_conversations", real_lookup)

        row = await Artifact.find_by_id_on_primary(image_id)
        assert row.conversation_id == kept
        assert row.key in fake.objects
        assert (await _served(async_client, token, image_id)).status_code == 200
    finally:
        await _cleanup(user)


@pytest.mark.asyncio
async def test_failed_put_leaves_no_row(async_client, monkeypatch):
    """Storage refuses the object: the request fails and no row exists."""
    fake = _use_fake_storage(monkeypatch)

    async def refuse(key, body, content_type):
        raise RuntimeError("put refused")

    monkeypatch.setattr(fake, "put_object", refuse)
    user, token = await create_test_user_and_token()
    try:
        with pytest.raises(RuntimeError):
            await _upload(async_client, token)
        assert await Artifact.count_documents({"user_id": user.id}) == 0
    finally:
        await _cleanup(user)


@pytest.mark.asyncio
async def test_disconnect_during_put_leaves_no_row(async_client, monkeypatch):
    """The request is cancelled while the object is written: no row is inserted."""
    fake = _use_fake_storage(monkeypatch)

    async def cancelled(key, body, content_type):
        raise asyncio.CancelledError()

    monkeypatch.setattr(fake, "put_object", cancelled)
    user, token = await create_test_user_and_token()
    try:
        with pytest.raises(BaseException):
            await _upload(async_client, token)
        assert await Artifact.count_documents({"user_id": user.id}) == 0
    finally:
        await _cleanup(user)


@pytest.mark.asyncio
async def test_rejected_type_leaves_no_row_and_no_object(async_client, monkeypatch):
    """Bytes that are not an allowed type: 415, nothing stored."""
    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.post(
            "/artifacts",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("pic.png", io.BytesIO(b"not an image at all"), "image/png")},
        )
        assert resp.status_code == 415
        assert fake.objects == {}
        assert await Artifact.count_documents({"user_id": user.id}) == 0
    finally:
        await _cleanup(user)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_conversation", [False, True])
async def test_read_path_finds_what_the_upload_wrote(async_client, monkeypatch, with_conversation):
    """The key the upload writes is the key the read serves, before and after an attach."""
    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        conv = await _conversation(user.id, "Layout") if with_conversation else None
        image_id = (await _upload(async_client, token, conv)).json()["id"]
        row = await Artifact.find_by_id_on_primary(image_id)
        assert list(fake.objects) == [row.key]
        served = await _served(async_client, token, image_id)
        assert served.status_code == 200 and served.content == PNG_BYTES

        target = await _conversation(user.id, "Target")
        await move_to_conversation(row, target)
        assert list(fake.objects) == [row.key]
        served = await _served(async_client, token, image_id)
        assert served.status_code == 200 and served.content == PNG_BYTES
    finally:
        await _cleanup(user)


@pytest.mark.asyncio
async def test_missing_object_is_a_quiet_404_without_the_key(async_client, monkeypatch, caplog):
    """A row whose bytes are gone answers 404; the log names the id only, below
    WARNING (the key carries the user's filename)."""
    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        conv = await _conversation(user.id, "Gone")
        image_id = (await _upload(async_client, token, conv)).json()["id"]
        row = await Artifact.find_by_id_on_primary(image_id)
        fake.objects.clear()

        with caplog.at_level(logging.DEBUG, logger="src.routers.artifact"):
            resp = await _served(async_client, token, image_id)
        assert resp.status_code == 404
        records = [r for r in caplog.records if r.name == "src.routers.artifact"]
        assert records, "the missing object is still recorded"
        assert all(r.levelno < logging.WARNING for r in records)
        assert all(row.key not in r.getMessage() for r in records)
        assert all("pic.png" not in r.getMessage() for r in records)
        assert any(image_id in r.getMessage() for r in records)
    finally:
        await _cleanup(user)
