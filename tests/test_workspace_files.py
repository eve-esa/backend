"""Workspace files: served with the tool allowlist, erased with their conversation."""

from urllib.parse import quote

import pytest

from src.database.models.conversation import Conversation
from src.services import artifact_cleanup
from src.services.agents.core.workspace import workspace_prefix, workspace_state_key
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
HTML_BYTES = b"<html><body><script>alert(1)</script></body></html>"
PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _use(monkeypatch, fake):
    monkeypatch.setattr("src.routers.conversation.storage_service", fake)
    monkeypatch.setattr("src.routers.artifact.storage_service", fake)
    monkeypatch.setattr("src.services.artifact_cleanup.storage_service", fake)
    return fake


async def _workspace_file(fake, user_id, conversation_id, relative, body, content_type):
    key = f"{workspace_prefix(user_id, conversation_id)}{relative}"
    await fake.put_object(key, body, content_type)
    return key


# ---------------- Served type -----------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative,body,stored_type",
    [
        ("report.html", HTML_BYTES, "text/html"),
        ("chart.svg", b"<svg><script>alert(1)</script></svg>", "image/svg+xml"),
        ("paper.pdf", PDF_BYTES, "application/pdf"),
    ],
)
async def test_a_stored_risky_type_is_served_opaque(
    async_client, monkeypatch, relative, body, stored_type
):
    """Objects saved before the type came from the bytes keep their old type."""
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Files")
    await _workspace_file(fake, user.id, conversation.id, relative, body, stored_type)
    try:
        resp = await async_client.get(
            f"/conversations/{conversation.id}/workspace/{relative}",
            headers=_auth(token),
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["content-disposition"] == f'attachment; filename="{relative}"'
        assert resp.headers["x-content-type-options"] == "nosniff"
    finally:
        await cleanup_models([user, conversation])


@pytest.mark.asyncio
async def test_a_png_stays_inline(async_client, monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Files")
    await _workspace_file(fake, user.id, conversation.id, "plots/ndvi.png", PNG_BYTES, "image/png")
    try:
        resp = await async_client.get(
            f"/conversations/{conversation.id}/workspace/plots/ndvi.png",
            headers=_auth(token),
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["content-disposition"] == 'inline; filename="ndvi.png"'
        assert resp.content == PNG_BYTES
    finally:
        await cleanup_models([user, conversation])


@pytest.mark.asyncio
async def test_a_non_latin1_filename_is_served_not_500(async_client, monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Files")
    name = "火灾报告.png"
    await _workspace_file(fake, user.id, conversation.id, name, PNG_BYTES, "image/png")
    try:
        resp = await async_client.get(
            f"/conversations/{conversation.id}/workspace/{quote(name)}",
            headers=_auth(token),
        )

        assert resp.status_code == 200
        assert f"filename*=UTF-8''{quote(name, safe='')}" in resp.headers["content-disposition"]
    finally:
        await cleanup_models([user, conversation])


# ---------------- Erasure -----------------


async def test_the_cascade_erases_the_workspace_and_its_state(monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    other_key = await _workspace_file(fake, "u1", "other", "keep.txt", b"keep", "text/plain")
    await _workspace_file(fake, "u1", "c1", "report.md", b"# hi", "application/octet-stream")
    await _workspace_file(fake, "u1", "c1", "plots/ndvi.png", PNG_BYTES, "image/png")
    await fake.put_object(workspace_state_key("u1", "c1"), b"{}", "application/json")

    await artifact_cleanup.delete_conversation_artifacts("u1", "c1")

    assert not [k for k in fake.objects if k.startswith(workspace_prefix("u1", "c1"))]
    assert workspace_state_key("u1", "c1") not in fake.objects
    assert other_key in fake.objects


async def test_a_capped_workspace_delete_reports_itself_incomplete(monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    monkeypatch.setattr(artifact_cleanup, "PREFIX_DELETE_MAX_KEYS", 1)
    await _workspace_file(fake, "u1", "c1", "a.txt", b"a", "text/plain")
    await _workspace_file(fake, "u1", "c1", "b.txt", b"b", "text/plain")
    await fake.put_object(workspace_state_key("u1", "c1"), b"{}", "application/json")

    with pytest.raises(artifact_cleanup.ArtifactCascadeIncomplete) as err:
        await artifact_cleanup.delete_conversation_artifacts("u1", "c1")

    assert err.value.failed == ["workspace"]
    assert workspace_state_key("u1", "c1") in fake.objects


async def test_conversation_delete_erases_the_workspace(async_client, monkeypatch):
    fake = _use(monkeypatch, FakeStorage())
    user, token = await create_test_user_and_token()
    conversation = await Conversation.create(user_id=user.id, name="Files")
    await _workspace_file(fake, user.id, conversation.id, "report.md", b"# hi", "text/plain")
    await fake.put_object(
        workspace_state_key(user.id, conversation.id), b"{}", "application/json"
    )
    try:
        resp = await async_client.delete(
            f"/conversations/{conversation.id}", headers=_auth(token)
        )

        assert resp.status_code == 200
        assert await Conversation.find_by_id(conversation.id) is None
        assert not fake.objects
    finally:
        await cleanup_models([user, conversation])
