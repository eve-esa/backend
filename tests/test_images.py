import io
import re
import uuid

import pytest

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models
from src.database.models.image import Image
from src.database.mongo import get_collection
from src.services.storage import StorageService, sniff_image_type


# Minimal valid magic-byte headers for the allowed image types.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF_BYTES = b"GIF89a" + b"\x00" * 64
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 64


class _FakeBody:
    """Stand-in for a boto3 StreamingBody used by the fake storage."""

    def __init__(self, data: bytes):
        self.data = data

    def iter_chunks(self, chunk_size):
        yield self.data

    def close(self):
        pass


class FakeStorage:
    """In-memory storage backend mirroring the StorageService interface."""

    def __init__(self):
        self.objects = {}

    def build_user_key(self, user_id: str, ext: str) -> str:
        return f"users/{user_id}/{uuid.uuid4().hex}.{ext.lstrip('.').lower()}"

    async def put_object(self, key, body, content_type):
        self.objects[key] = {"body": body, "content_type": content_type}

    async def get_object(self, key):
        obj = self.objects[key]
        return {
            "Body": _FakeBody(obj["body"]),
            "ETag": '"fake-etag"',
            "ContentLength": len(obj["body"]),
        }

    async def stream_body(self, body, chunk_size=65536):
        yield body.data

    async def delete_object(self, key):
        self.objects.pop(key, None)

    async def generate_presigned_get(self, key, expires_in=None):
        return f"http://minio.local/{key}?signed=1"

    async def generate_presigned_put(self, key, content_type=None, expires_in=None):
        return f"http://minio.local/{key}?signed=1"


def _use_fake_storage(monkeypatch) -> FakeStorage:
    """Monkeypatch the router's storage singleton with an in-memory fake."""
    fake = FakeStorage()
    monkeypatch.setattr("src.routers.image.storage_service", fake)
    return fake


async def _upload(async_client, token, filename, data, content_type):
    return await async_client.post(
        "/images",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, io.BytesIO(data), content_type)},
    )


async def _cleanup_quota(user_id):
    await get_collection("image_upload_quota").delete_many({"user_id": user_id})


# ---------------- Upload -----------------


@pytest.mark.asyncio
async def test_upload_valid_png(async_client, monkeypatch):
    """A valid PNG is stored under the per-user key and returns embed metadata."""

    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        assert resp.status_code == 200
        body = resp.json()
        assert body["url"] == f"/images/{body['id']}"
        assert body["markdown"] == f"![pic.png](/images/{body['id']})"
        assert body["filename"] == "pic.png"
        assert body["content_type"] == "image/png"
        assert body["size_bytes"] == len(PNG_BYTES)

        image = await Image.find_by_id(body["id"])
        assert image is not None
        assert re.match(rf"^users/{user.id}/[0-9a-f]{{32}}\.png$", image.key)
        assert image.key in fake.objects
    finally:
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_spoofed_content_type_rejected(async_client, monkeypatch):
    """A non-image payload declaring an image Content-Type is rejected (415)."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(
            async_client, token, "evil.png", b"this is not an image", "image/png"
        )
        assert resp.status_code == 415
    finally:
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_oversize_rejected(async_client, monkeypatch):
    """A payload larger than IMAGE_MAX_BYTES is rejected (413)."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.image.IMAGE_MAX_BYTES", 16)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(
            async_client, token, "big.png", PNG_BYTES, "image/png"
        )
        assert resp.status_code == 413
    finally:
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_quota_exceeded(async_client, monkeypatch):
    """Reaching the daily quota returns 429."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.image.IMAGE_UPLOADS_PER_DAY", 0)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        assert resp.status_code == 429
    finally:
        await _cleanup_quota(user.id)
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_quota_enforced_by_atomic_counter(async_client, monkeypatch):
    """The atomic per-day counter caps uploads: the 2nd upload with limit=1 is 429."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.image.IMAGE_UPLOADS_PER_DAY", 1)
    user, token = await create_test_user_and_token()
    try:
        first = await _upload(async_client, token, "a.png", PNG_BYTES, "image/png")
        assert first.status_code == 200
        second = await _upload(async_client, token, "b.png", PNG_BYTES, "image/png")
        assert second.status_code == 429
    finally:
        await _cleanup_quota(user.id)
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- Serving -----------------


@pytest.mark.asyncio
async def test_get_image_owner(async_client, monkeypatch):
    """The owner can fetch the image bytes with the sniffed media type."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        image_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        resp = await async_client.get(
            f"/images/{image_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        assert resp.content == PNG_BYTES
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["cache-control"] == "private, max-age=3600"
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["content-disposition"] == 'inline; filename="pic.png"'
    finally:
        await _cleanup_quota(user.id)
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_get_image_other_user_forbidden(async_client, monkeypatch):
    """A different user cannot fetch someone else's image (403)."""

    _use_fake_storage(monkeypatch)
    owner, owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        image_id = (
            await _upload(async_client, owner_token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        resp = await async_client.get(
            f"/images/{image_id}", headers={"Authorization": f"Bearer {intr_token}"}
        )
        assert resp.status_code == 403
    finally:
        await Image.delete_many({"user_id": owner.id})
        await cleanup_models([owner, intruder])


@pytest.mark.asyncio
async def test_get_image_anonymous_forbidden(async_client, monkeypatch):
    """An unauthenticated request cannot fetch an image."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        image_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        resp = await async_client.get(f"/images/{image_id}")
        assert resp.status_code in (401, 403)
    finally:
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_get_image_missing(async_client, monkeypatch):
    """Fetching a non-existent image returns 404."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.get(
            "/images/000000000000000000000000",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
    finally:
        await cleanup_models([user])


# ---------------- List / delete -----------------


@pytest.mark.asyncio
async def test_list_and_delete_image(async_client, monkeypatch):
    """Listing returns the user's images; delete removes it from storage and DB."""

    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        image_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]
        key = (await Image.find_by_id(image_id)).key

        listing = await async_client.get(
            "/images", headers={"Authorization": f"Bearer {token}"}
        )
        assert listing.status_code == 200
        ids = [item["id"] for item in listing.json()["data"]]
        assert image_id in ids

        del_resp = await async_client.delete(
            f"/images/{image_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert del_resp.status_code == 200
        assert await Image.find_by_id(image_id) is None
        assert key not in fake.objects
    finally:
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_delete_image_storage_failure_keeps_record(async_client, monkeypatch):
    """If the object-store delete fails, the DB record is kept (fail-closed, no orphan)."""

    fake = _use_fake_storage(monkeypatch)

    async def _boom(key):
        raise RuntimeError("s3 unavailable")

    monkeypatch.setattr(fake, "delete_object", _boom)

    user, token = await create_test_user_and_token()
    try:
        image_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        del_resp = await async_client.delete(
            f"/images/{image_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert del_resp.status_code == 500

        # The record must survive so a retry can still reclaim the object.
        assert await Image.find_by_id(image_id) is not None
        get_resp = await async_client.get(
            f"/images/{image_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert get_resp.status_code == 200
    finally:
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- Chat wiring -----------------


async def _mock_generate_answer(request, conversation_id=None):
    return "Test answer", [], False, {}, {}, []


@pytest.mark.asyncio
async def test_message_with_image_ids_persists_attachments(async_client, monkeypatch):
    """image_ids on a message persist attachments and backfill conversation_id."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.message.generate_answer", _mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        image_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        conv_id = (
            await async_client.post(
                "/conversations",
                json={"name": "Image Conv"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        msg_resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json={"query": "look", "image_ids": [image_id]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert msg_resp.status_code == 200

        detail = await async_client.get(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert detail.status_code == 200
        messages = detail.json()["messages"]
        assert len(messages) == 1
        attachments = messages[0]["attachments"]
        assert attachments and attachments[0]["image_id"] == image_id
        assert attachments[0]["url"] == f"/images/{image_id}"
        assert attachments[0]["content_type"] == "image/png"

        # conversation_id was backfilled on the image
        image = await Image.find_by_id(image_id)
        assert image.conversation_id == conv_id

        await async_client.delete(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_message_with_other_users_image_forbidden(async_client, monkeypatch):
    """Attaching another user's image to a message is forbidden (403)."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.message.generate_answer", _mock_generate_answer)

    owner, owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        image_id = (
            await _upload(async_client, owner_token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        conv_id = (
            await async_client.post(
                "/conversations",
                json={"name": "Intruder Conv"},
                headers={"Authorization": f"Bearer {intr_token}"},
            )
        ).json()["id"]

        resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json={"query": "look", "image_ids": [image_id]},
            headers={"Authorization": f"Bearer {intr_token}"},
        )
        assert resp.status_code == 403

        await async_client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {intr_token}"},
        )
    finally:
        await Image.delete_many({"user_id": owner.id})
        await cleanup_models([owner, intruder])


@pytest.mark.asyncio
async def test_message_image_ids_over_cap_rejected(async_client, monkeypatch):
    """More than the allowed number of image_ids fails validation (422)."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.message.generate_answer", _mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        conv_id = (
            await async_client.post(
                "/conversations",
                json={"name": "Cap Conv"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json={"query": "look", "image_ids": [f"id{i}" for i in range(21)]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

        await async_client.delete(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_message_image_ids_deduped(async_client, monkeypatch):
    """Duplicate image_ids collapse to a single attachment."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.message.generate_answer", _mock_generate_answer)

    user, token = await create_test_user_and_token()
    try:
        image_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        conv_id = (
            await async_client.post(
                "/conversations",
                json={"name": "Dedupe Conv"},
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["id"]

        msg_resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json={"query": "look", "image_ids": [image_id, image_id, image_id]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert msg_resp.status_code == 200

        detail = await async_client.get(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
        attachments = detail.json()["messages"][0]["attachments"]
        assert len(attachments) == 1
        assert attachments[0]["image_id"] == image_id

        await async_client.delete(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await _cleanup_quota(user.id)
        await Image.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- Unit tests -----------------


def test_sniff_image_type():
    assert sniff_image_type(PNG_BYTES) == "png"
    assert sniff_image_type(JPEG_BYTES) == "jpeg"
    assert sniff_image_type(GIF_BYTES) == "gif"
    assert sniff_image_type(GIF_BYTES.replace(b"GIF89a", b"GIF87a", 1)) == "gif"
    assert sniff_image_type(WEBP_BYTES) == "webp"
    assert sniff_image_type(b"not an image") is None
    assert sniff_image_type(b"") is None


def test_build_user_key():
    key = StorageService.build_user_key("user123", "png")
    assert re.match(r"^users/user123/[0-9a-f]{32}\.png$", key)
    # extension is normalized (leading dot stripped, lowercased)
    assert StorageService.build_user_key("u", ".JPEG").endswith(".jpeg")
