"""Artifact serving: opaque type for anything outside the inline allowlist,
RFC 6266 filenames, 404 for foreign or missing ids, attach on the primary."""

from urllib.parse import quote

import pytest
from bson import ObjectId

from src.database.models.artifact import Artifact, ArtifactSource
from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.routers.artifact import content_disposition
from src.services.mcp.artifact_ingestion import _provenance_title, artifact_markdown_line
from src.services.storage import INLINE_CONTENT_TYPES, INLINE_IMAGE_CONTENT_TYPES
from tests.test_artifacts import (
    PNG_BYTES,
    _cleanup_quota,
    _mock_generate_answer,
    _upload,
    _use_fake_storage,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

SVG_BYTES = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
HTML_BYTES = b"<html><body><script>alert(1)</script></body></html>"


async def _seed(fake, user_id, content_type, body, filename):
    """A row stored before the type came from the bytes (MCP or display)."""
    key = fake.build_user_key(user_id, "bin", prefix="artifacts")
    await fake.put_object(key, body, content_type)
    return await Artifact.create(
        user_id=user_id,
        key=key,
        filename=filename,
        content_type=content_type,
        size_bytes=len(body),
        source=ArtifactSource(type="mcp_tool", mcp_server="srv", tool_name="plot"),
    )


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------- Served type -----------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type,body,filename",
    [
        ("image/svg+xml", SVG_BYTES, "chart.svg"),
        ("text/html", HTML_BYTES, "report.html"),
    ],
)
async def test_stored_svg_and_html_are_served_opaque(
    async_client, monkeypatch, content_type, body, filename
):
    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact = await _seed(fake, user.id, content_type, body, filename)

        resp = await async_client.get(f"/artifacts/{artifact.id}", headers=_auth(token))

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["content-disposition"] == f'attachment; filename="{filename}"'
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.content == body
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_png_keeps_its_type_inline(async_client, monkeypatch):
    _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]

        resp = await async_client.get(f"/artifacts/{artifact_id}", headers=_auth(token))

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["content-disposition"] == 'inline; filename="pic.png"'
    finally:
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_presigned_url_signs_the_opaque_type_for_svg(async_client, monkeypatch):
    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact = await _seed(fake, user.id, "image/svg+xml", SVG_BYTES, "chart.svg")

        resp = await async_client.get(f"/artifacts/{artifact.id}/url", headers=_auth(token))

        assert resp.status_code == 200
        assert fake.last_presign == {
            "key": artifact.key,
            "response_content_type": "application/octet-stream",
            "response_content_disposition": 'attachment; filename="chart.svg"',
        }
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_presigned_url_keeps_png_inline(async_client, monkeypatch):
    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact = await _seed(fake, user.id, "image/png", PNG_BYTES, "pic.png")

        resp = await async_client.get(f"/artifacts/{artifact.id}/url", headers=_auth(token))

        assert resp.status_code == 200
        assert fake.last_presign["response_content_type"] == "image/png"
        assert fake.last_presign["response_content_disposition"] == 'inline; filename="pic.png"'
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


def test_one_allowlist_feeds_both_the_router_and_the_stub():
    assert INLINE_IMAGE_CONTENT_TYPES <= INLINE_CONTENT_TYPES
    assert "image/svg+xml" not in INLINE_CONTENT_TYPES
    assert "text/html" not in INLINE_CONTENT_TYPES


# ---------------- Filenames (RFC 6266) -----------------


def test_ascii_name_keeps_the_plain_form():
    assert content_disposition("inline", "pic.png") == 'inline; filename="pic.png"'


@pytest.mark.parametrize("name", ["报告.pdf", "rapport été.csv", "plot 📈.png"])
def test_non_latin1_name_gets_a_fallback_and_an_encoded_form(name):
    value = content_disposition("attachment", name)

    value.encode("latin-1")  # what Starlette does with every header value
    assert value.isascii()
    assert value.endswith(f"filename*=UTF-8''{quote(name, safe='')}")
    fallback = value.split('filename="', 1)[1].split('"', 1)[0]
    assert fallback.isascii() and fallback.isprintable()


def test_accents_fold_in_the_fallback():
    value = content_disposition("attachment", "café.txt")
    assert 'filename="cafe.txt"' in value


@pytest.mark.parametrize(
    "name", ['a"b.txt', "a\\b.txt", "../a/b.txt", "a\r\nX-Evil: 1.txt", "a\x01\x7fb.txt"]
)
def test_unsafe_characters_never_reach_the_header(name):
    value = content_disposition("attachment", name)
    plain = value.split('filename="', 1)[1].rsplit('"', 1)[0]
    for char in '"\\/\r\n\x01\x7f':
        assert char not in plain


def test_empty_name_falls_back():
    assert content_disposition("inline", "") == 'inline; filename="artifact"'
    assert content_disposition("inline", None) == 'inline; filename="artifact"'


@pytest.mark.asyncio
async def test_cjk_filename_is_served_not_500(async_client, monkeypatch):
    fake = _use_fake_storage(monkeypatch)
    user, token = await create_test_user_and_token()
    try:
        artifact = await _seed(fake, user.id, "image/png", PNG_BYTES, "图表 1.png")

        resp = await async_client.get(f"/artifacts/{artifact.id}", headers=_auth(token))

        assert resp.status_code == 200
        disposition = resp.headers["content-disposition"]
        assert disposition.startswith('inline; filename="')
        assert f"filename*=UTF-8''{quote('图表 1.png', safe='')}" in disposition
    finally:
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- 404 for foreign or missing ids -----------------


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", "/url"])
async def test_foreign_and_missing_ids_answer_404(async_client, monkeypatch, suffix):
    fake = _use_fake_storage(monkeypatch)
    owner, _owner_token = await create_test_user_and_token()
    intruder, intr_token = await create_test_user_and_token()
    try:
        artifact = await _seed(fake, owner.id, "image/png", PNG_BYTES, "pic.png")

        foreign = await async_client.get(
            f"/artifacts/{artifact.id}{suffix}", headers=_auth(intr_token)
        )
        missing = await async_client.get(
            f"/artifacts/{ObjectId()}{suffix}", headers=_auth(intr_token)
        )

        assert foreign.status_code == 404
        assert missing.status_code == 404
        assert foreign.json() == missing.json()
    finally:
        await Artifact.delete_many({"user_id": owner.id})
        await cleanup_models([owner, intruder])


@pytest.mark.asyncio
async def test_attach_missing_id_answers_404(async_client, monkeypatch):
    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.message.generate_answer", _mock_generate_answer)
    user, token = await create_test_user_and_token()
    try:
        conv_id = (
            await async_client.post("/conversations", json={"name": "c"}, headers=_auth(token))
        ).json()["id"]

        resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json={"query": "look", "artifact_ids": [str(ObjectId())]},
            headers=_auth(token),
        )

        assert resp.status_code == 404
        await async_client.delete(f"/conversations/{conv_id}", headers=_auth(token))
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_attach_reads_the_primary_right_after_an_upload(async_client, monkeypatch):
    """A secondary that has not caught up must not turn a fresh upload into 404."""
    _use_fake_storage(monkeypatch)
    monkeypatch.setattr("src.routers.message.generate_answer", _mock_generate_answer)
    user, token = await create_test_user_and_token()
    conv_id = None
    try:
        image_id = (
            await _upload(async_client, token, "pic.png", PNG_BYTES, "image/png")
        ).json()["id"]
        conv_id = (
            await async_client.post("/conversations", json={"name": "c"}, headers=_auth(token))
        ).json()["id"]

        async def lagging_secondary(_cls, _artifact_id):
            return None

        monkeypatch.setattr(Artifact, "find_by_id", classmethod(lagging_secondary))

        resp = await async_client.post(
            f"/conversations/{conv_id}/messages",
            json={"query": "look", "artifact_ids": [image_id]},
            headers=_auth(token),
        )

        assert resp.status_code == 200
    finally:
        await Message.delete_many({"conversation_id": conv_id})
        await Conversation.delete_many({"user_id": user.id})
        await _cleanup_quota(user.id)
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


# ---------------- Stub line -----------------


def test_svg_gets_a_link_never_an_embed():
    line = artifact_markdown_line(
        artifact_id="abc", filename="chart.svg", content_type="image/svg+xml"
    )
    assert line == '[chart.svg](/artifacts/abc "MCP")'


def test_png_gets_an_embed():
    line = artifact_markdown_line(
        artifact_id="abc",
        filename="pic.png",
        content_type="image/png",
        server_name="srv",
        tool_name="plot",
    )
    assert line == '![pic.png](/artifacts/abc "MCP: srv/plot")'


@pytest.mark.parametrize(
    "server,tool",
    [("evil\n\nserver", "plot"), ("srv", 'to"ol\\'), ("srv\\", "plot")],
)
def test_provenance_title_cannot_break_out(server, tool):
    title = _provenance_title(server, tool)
    assert "\n" not in title
    assert '"' not in title
    assert "\\" not in title
    assert title.startswith("MCP: ")
