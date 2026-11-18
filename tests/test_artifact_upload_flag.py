"""FEATURE_ARTIFACT_UPLOADS: off by default, and off answers a real 404."""

import importlib
import io

import pytest

from src.database.models.artifact import Artifact
from src.database.mongo import get_collection
from tests.utils.cleaner import cleanup_models
from tests.utils.fake_storage import FakeStorage
from tests.utils.utils import create_test_user_and_token

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


async def _post_upload(async_client, token):
    return await async_client.post(
        "/artifacts",
        headers=_auth(token),
        files={"file": ("pic.png", io.BytesIO(PNG_BYTES), "image/png")},
    )


async def test_upload_off_answers_404_and_stores_nothing(async_client, monkeypatch):
    fake = FakeStorage()
    monkeypatch.setattr("src.routers.artifact.storage_service", fake)
    monkeypatch.setattr("src.config.FEATURE_ARTIFACT_UPLOADS", False)
    user, token = await create_test_user_and_token()
    try:
        resp = await _post_upload(async_client, token)

        # 404, not 405: GET /artifacts shares the path and must not give it away.
        assert resp.status_code == 404
        assert not fake.objects
        assert await Artifact.count_documents({"user_id": user.id}) == 0
        assert await get_collection("image_upload_quota").find_one({"user_id": user.id}) is None

        # Reading and listing stay open.
        listed = await async_client.get("/artifacts", headers=_auth(token))
        assert listed.status_code == 200
    finally:
        await cleanup_models([user])


async def test_upload_on_answers_200(async_client, monkeypatch):
    monkeypatch.setattr("src.routers.artifact.storage_service", FakeStorage())
    monkeypatch.setattr("src.config.FEATURE_ARTIFACT_UPLOADS", True)
    user, token = await create_test_user_and_token()
    try:
        resp = await _post_upload(async_client, token)
        assert resp.status_code == 200, resp.text
    finally:
        await get_collection("image_upload_quota").delete_many({"user_id": user.id})
        await Artifact.delete_many({"user_id": user.id})
        await cleanup_models([user])


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("", False), (" ", False), ("false", False), ("true", True), ("TRUE", True)],
)
def test_upload_flag_defaults_off(monkeypatch, value, expected):
    import src.config as config

    # A local .env (compose gate) must not decide the default, nor leak into
    # os.environ for the tests that follow.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    if value is None:
        monkeypatch.delenv("FEATURE_ARTIFACT_UPLOADS", raising=False)
    else:
        monkeypatch.setenv("FEATURE_ARTIFACT_UPLOADS", value)
    try:
        assert importlib.reload(config).FEATURE_ARTIFACT_UPLOADS is expected
    finally:
        monkeypatch.delenv("FEATURE_ARTIFACT_UPLOADS", raising=False)
        importlib.reload(config)
        monkeypatch.undo()
