import io
import re

import pytest
from botocore.exceptions import ClientError

from tests.utils.utils import create_test_user_and_token
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from src.database.models.artifact import Artifact, ArtifactSource
from src.database.mongo import get_collection
from src.services.storage import (
    ObjectNotFoundError,
    StorageService,
    sniff_image_type,
)


# Minimal valid magic-byte headers for the allowed image types.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF_BYTES = b"GIF89a" + b"\x00" * 64
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 64

# Minimal valid payloads for the generalized non-image artifact types.
PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n" + b"0 0 obj\n<< >>\nendobj\n"
CSV_TEXT = b"name,value\nfoo,1\nbar,2\n"
TXT_TEXT = b"just a plain text file\n"
JSON_TEXT = b'{"hello": "world"}'
GEOJSON_TEXT = b'{"type": "FeatureCollection", "features": []}'


def _use_fake_storage(monkeypatch) -> FakeStorage:
    """Monkeypatch the router's storage singleton with an in-memory fake."""
    fake = FakeStorage()
    monkeypatch.setattr("src.routers.artifact.storage_service", fake)
    return fake


# The full generalized allowlist, set explicitly by tests that exercise the
# non-image types: the ambient ARTIFACT_UPLOAD_ALLOWED_TYPES may resolve to
# the legacy image-only IMAGE_ALLOWED_TYPES value depending on which env vars
# happen to be set in the environment the tests run in (see config.py's
# fallback chain), so tests must not rely on it.
FULL_ARTIFACT_ALLOWED_TYPES = ["png", "jpeg", "webp", "gif", "pdf", "csv", "txt", "json", "geojson"]


def _use_full_artifact_allowlist(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.routers.artifact.ARTIFACT_UPLOAD_ALLOWED_TYPES", FULL_ARTIFACT_ALLOWED_TYPES
    )


async def _upload(async_client, token, filename, data, content_type):
    return await async_client.post(
        "/artifacts",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, io.BytesIO(data), content_type)},
    )


async def _cleanup_quota(user_id):
    await get_collection("image_upload_quota").delete_many({"user_id": user_id})


async def _seed_mcp_artifact(fake: FakeStorage, user_id: str, content_type: str, body: bytes = b"tool output") -> Artifact:
    """Seed an artifact as if produced by an MCP tool call (not through the upload endpoint)."""
    key = fake.build_user_key(user_id, "bin", prefix="artifacts")
    await fake.put_object(key, body, content_type)
    return await Artifact.create(
        user_id=user_id,
        key=key,
        filename="tool-output.bin",
        content_type=content_type,
        size_bytes=len(body),
        source=ArtifactSource(type="mcp_tool", mcp_server="wiley", tool_name="search"),
    )


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
        assert body["url"] == f"/artifacts/{body['id']}"
        assert body["markdown"] == f"![pic.png](/artifacts/{body['id']})"
        assert body["filename"] == "pic.png"
        assert body["content_type"] == "image/png"
        assert body["size_bytes"] == len(PNG_BYTES)

        artifact = await Artifact.find_by_id(body["id"])
        assert artifact is not None
        assert re.match(
            rf"^users/{user.id}/artifacts/[0-9a-f]{{32}}\.png$", artifact.key
        )
        assert artifact.key in fake.objects
        assert artifact.source.type == "upload"
    finally:
        await Artifact.delete_many({"user_id": user.id})
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
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_valid_pdf(async_client, monkeypatch):
    """A valid PDF is accepted via its magic prefix, regardless of the declared Content-Type."""

    _use_fake_storage(monkeypatch)
    _use_full_artifact_allowlist(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(
            async_client, token, "doc.pdf", PDF_BYTES, "application/octet-stream"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["content_type"] == "application/pdf"
        assert body["size_bytes"] == len(PDF_BYTES)
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.parametrize(
    "filename,data,expected_content_type",
    [
        ("data.csv", CSV_TEXT, "text/csv"),
        ("notes.txt", TXT_TEXT, "text/plain"),
        ("config.json", JSON_TEXT, "application/json"),
        ("shape.geojson", GEOJSON_TEXT, "application/geo+json"),
    ],
)
@pytest.mark.asyncio
async def test_upload_valid_text_like_types(
    async_client, monkeypatch, filename, data, expected_content_type
):
    """Text-like types are accepted when the extension matches a decodable UTF-8 payload."""

    _use_fake_storage(monkeypatch)
    _use_full_artifact_allowlist(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(async_client, token, filename, data, "application/octet-stream")
        assert resp.status_code == 200
        body = resp.json()
        assert body["content_type"] == expected_content_type
        assert body["size_bytes"] == len(data)
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_extension_content_mismatch_rejected(async_client, monkeypatch):
    """A .csv extension over undecodable binary content is rejected (415), not smuggled through."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(
            async_client, token, "data.csv", b"\x00\x01\xff\xfe binary", "text/csv"
        )
        assert resp.status_code == 415
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_unrecognized_extension_rejected(async_client, monkeypatch):
    """A file with no magic bytes and an extension outside the text-like allowlist is rejected (415)."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(
            async_client, token, "script.exe", b"MZ\x90\x00", "application/octet-stream"
        )
        assert resp.status_code == 415
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_oversize_rejected(async_client, monkeypatch):
    """A payload larger than ARTIFACT_UPLOAD_MAX_BYTES is rejected (413)."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.artifact.ARTIFACT_UPLOAD_MAX_BYTES", 16)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(
            async_client, token, "big.png", PNG_BYTES, "image/png"
        )
        assert resp.status_code == 413
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_quota_exceeded(async_client, monkeypatch):
    """Reaching the daily quota returns 429."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.artifact.ARTIFACT_UPLOADS_PER_DAY", 0)
    user, token = await create_test_user_and_token()
    try:
        resp = await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        assert resp.status_code == 429
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_upload_quota_enforced_by_atomic_counter(async_client, monkeypatch):
    """The atomic per-day counter caps uploads: the 2nd upload with limit=1 is 429."""

    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.artifact.ARTIFACT_UPLOADS_PER_DAY", 1)
    user, token = await create_test_user_and_token()
    try:
        first = await _upload(async_client, token, "a.png", PNG_BYTES, "image/png")
        assert first.status_code == 200
        second = await _upload(async_client, token, "b.png", PNG_BYTES, "image/png")
        assert second.status_code == 429
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- Serving -----------------


@pytest.mark.asyncio
async def test_get_artifact_owner(async_client, monkeypatch):
    """The owner can fetch the artifact bytes with the sniffed media type."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        resp = await async_client.get(
            f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        assert resp.content == PNG_BYTES
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["cache-control"] == "private, no-cache"
        # Vary: Authorization keys any HTTP cache on the token, preventing a
        # different user from being served another user's cached bytes.
        assert resp.headers["vary"] == "Authorization"
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["content-disposition"] == 'inline; filename="pic.png"'
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_get_artifact_other_user_forbidden(async_client, monkeypatch):
    """A different user cannot fetch someone else's artifact (403)."""

    _use_fake_storage(monkeypatch)
    owner, owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(async_client, owner_token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        resp = await async_client.get(
            f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {intr_token}"}
        )
        assert resp.status_code == 403
    finally:
        await Artifact.delete_many({"user_id": owner.id})
        await cleanup_models([owner, intruder])


@pytest.mark.asyncio
async def test_get_artifact_anonymous_forbidden(async_client, monkeypatch):
    """An unauthenticated request cannot fetch an artifact."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        resp = await async_client.get(f"/artifacts/{artifact_id}")
        assert resp.status_code in (401, 403)
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_get_artifact_missing(async_client, monkeypatch):
    """Fetching a non-existent artifact returns 404."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        resp = await async_client.get(
            "/artifacts/000000000000000000000000",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_get_artifact_object_missing_from_storage(async_client, monkeypatch):
    """A record whose bytes are gone returns 404, not an unhandled 500."""

    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        # Drop the bytes but keep the Mongo record, reproducing the drift seen
        # in production (lifecycle rule, manual cleanup, half-failed upload).
        fake.objects.clear()

        resp = await async_client.get(
            f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Artifact content is no longer available"
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_storage_get_object_translates_no_such_key(monkeypatch):
    """StorageService turns botocore's NoSuchKey into ObjectNotFoundError."""

    service = StorageService()

    class _Boom:
        def get_object(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "The key does not exist"}},
                "GetObject",
            )

    monkeypatch.setattr(service, "_client", lambda: _Boom())
    with pytest.raises(ObjectNotFoundError):
        await service.get_object("users/u1/artifacts/gone.png")


@pytest.mark.no_db
@pytest.mark.asyncio
async def test_storage_get_object_reraises_other_client_errors(monkeypatch):
    """A permissions failure is not silently reported as a missing object."""

    service = StorageService()

    class _Denied:
        def get_object(self, **kwargs):
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
                "GetObject",
            )

    monkeypatch.setattr(service, "_client", lambda: _Denied())
    with pytest.raises(ClientError):
        await service.get_object("users/u1/artifacts/forbidden.png")


# ---------------- Disposition policy -----------------


@pytest.mark.asyncio
async def test_disposition_inline_for_image(async_client, monkeypatch):
    """Image types render inline."""

    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        resp = await async_client.get(
            f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.headers["content-disposition"].startswith("inline")
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_disposition_attachment_for_html(async_client, monkeypatch):
    """text/html is forced to attachment (stored-XSS defense), even for an owner."""

    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact = await _seed_mcp_artifact(fake, user.id, "text/html", b"<script>1</script>")

        resp = await async_client.get(
            f"/artifacts/{artifact.id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("attachment")
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_disposition_attachment_for_unknown_type(async_client, monkeypatch):
    """An unrecognized/unknown content type is forced to attachment."""

    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact = await _seed_mcp_artifact(
            fake, user.id, "application/octet-stream", b"\x00\x01\x02"
        )

        resp = await async_client.get(
            f"/artifacts/{artifact.id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("attachment")
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_disposition_attachment_for_csv(async_client, monkeypatch):
    """text/csv is forced to attachment, never inline as text/html (anti-XSS posture)."""

    _use_fake_storage(monkeypatch)
    _use_full_artifact_allowlist(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(async_client, token, "data.csv", CSV_TEXT, "text/csv")
        ).json()["id"]

        resp = await async_client.get(
            f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("attachment")
        assert resp.headers["x-content-type-options"] == "nosniff"
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_disposition_attachment_for_geojson(async_client, monkeypatch):
    """application/geo+json is forced to attachment (outside the inline allowlist)."""

    _use_fake_storage(monkeypatch)
    _use_full_artifact_allowlist(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(
                async_client, token, "shape.geojson", GEOJSON_TEXT, "application/geo+json"
            )
        ).json()["id"]

        resp = await async_client.get(
            f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("attachment")
        assert resp.headers["x-content-type-options"] == "nosniff"
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_disposition_inline_for_pdf(async_client, monkeypatch):
    """application/pdf renders inline (already in the safe-types allowlist)."""

    _use_fake_storage(monkeypatch)
    _use_full_artifact_allowlist(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(
                async_client, token, "doc.pdf", PDF_BYTES, "application/octet-stream"
            )
        ).json()["id"]

        resp = await async_client.get(
            f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].startswith("inline")
        assert resp.headers["x-content-type-options"] == "nosniff"
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- List / delete -----------------


@pytest.mark.asyncio
async def test_list_artifacts(async_client, monkeypatch):
    """Listing returns the user's artifacts, with their source."""

    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]
        listing = await async_client.get(
            "/artifacts", headers={"Authorization": f"Bearer {token}"}
        )
        assert listing.status_code == 200
        items = listing.json()["data"]
        ids = [item["id"] for item in items]
        assert artifact_id in ids
        # source is included in the list payload.
        matched = next(item for item in items if item["id"] == artifact_id)
        assert matched["source"]["type"] == "upload"
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_list_artifacts_filtered_by_conversation(async_client, monkeypatch):
    """The conversation_id query filters the listing to that conversation only."""

    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        in_conv_id = (
            await _upload(async_client, token, "a.png", PNG_BYTES, "image/png")
        ).json()["id"]
        other_id = (
            await _upload(async_client, token, "b.png", PNG_BYTES, "image/png")
        ).json()["id"]

        artifact = await Artifact.find_by_id(in_conv_id)
        artifact.conversation_id = "conv-1"
        await artifact.save()

        other = await Artifact.find_by_id(other_id)
        other.conversation_id = "conv-2"
        await other.save()

        listing = await async_client.get(
            "/artifacts?conversation_id=conv-1",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert listing.status_code == 200
        ids = [item["id"] for item in listing.json()["data"]]
        assert ids == [in_conv_id]
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_artifacts_cannot_be_deleted(async_client, monkeypatch):
    """An uploaded artifact is permanent: there is no endpoint to remove it.

    This replaces three tests that exercised DELETE /artifacts/{id} — the happy
    path, the storage-failure path that kept the record, and the 403 that spared
    MCP artifacts. None of that has to be right any more, because the method is
    not served: once a file is uploaded it stays, which is how the products this
    one is measured against behave.

    405 rather than 404 is the point. The path still exists for GET, so FastAPI
    answers "method not allowed" — that is what tells a client the resource is
    real and the operation is gone, instead of leaving it to guess.
    """

    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]
        key = (await Artifact.find_by_id(artifact_id)).key

        del_resp = await async_client.delete(
            f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert del_resp.status_code == 405

        # Not merely refused: still there, in the database and in the object store,
        # and still readable. A 405 with the bytes gone would be worse than a 200.
        assert await Artifact.find_by_id(artifact_id) is not None
        assert key in fake.objects
        get_resp = await async_client.get(
            f"/artifacts/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert get_resp.status_code == 200
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- Chat wiring -----------------


async def _mock_generate_answer(request, conversation_id=None, user_id=None):
    return "Test answer", [], False, {}, {}, []


@pytest.mark.asyncio
async def test_message_with_artifact_ids_persists_attachments(async_client, monkeypatch):
    """artifact_ids on a message persist attachments and backfill conversation_id."""

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
            json={"query": "look", "artifact_ids": [image_id]},
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
        assert attachments[0]["url"] == f"/artifacts/{image_id}"
        assert attachments[0]["content_type"] == "image/png"
        assert attachments[0]["source"]["type"] == "upload"

        # conversation_id was backfilled on the artifact
        artifact = await Artifact.find_by_id(image_id)
        assert artifact.conversation_id == conv_id

        await async_client.delete(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_message_with_legacy_image_ids_alias_accepted(async_client, monkeypatch):
    """The legacy ``image_ids`` field name is still accepted as an alias for artifact_ids."""

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
                json={"name": "Legacy Alias Conv"},
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
        attachments = detail.json()["messages"][0]["attachments"]
        assert attachments and attachments[0]["image_id"] == image_id

        await async_client.delete(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_message_with_other_users_image_forbidden(async_client, monkeypatch):
    """Attaching another user's artifact to a message is forbidden (403)."""

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
            json={"query": "look", "artifact_ids": [image_id]},
            headers={"Authorization": f"Bearer {intr_token}"},
        )
        assert resp.status_code == 403

        await async_client.delete(
            f"/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {intr_token}"},
        )
    finally:
        await Artifact.delete_many({"user_id": owner.id})
        await cleanup_models([owner, intruder])


@pytest.mark.asyncio
async def test_message_artifact_ids_over_cap_rejected(async_client, monkeypatch):
    """More than the allowed number of artifact_ids fails validation (422)."""

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
            json={"query": "look", "artifact_ids": [f"id{i}" for i in range(21)]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

        await async_client.delete(
            f"/conversations/{conv_id}", headers={"Authorization": f"Bearer {token}"}
        )
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_message_artifact_ids_deduped(async_client, monkeypatch):
    """Duplicate artifact_ids collapse to a single attachment."""

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
            json={"query": "look", "artifact_ids": [image_id, image_id, image_id]},
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
        await Artifact.delete_many({"user_id": user.id})
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


def test_build_user_key_with_prefix():
    key = StorageService.build_user_key("user123", "bin", prefix="artifacts")
    assert re.match(r"^users/user123/artifacts/[0-9a-f]{32}\.bin$", key)
