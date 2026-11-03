"""GET /skills lists a skill and its files, never the file bytes."""

from datetime import datetime, timezone

import pytest

from src.database.models.skill import Skill, SkillFile
from tests.utils.cleaner import cleanup_models
from tests.utils.utils import create_test_user_and_token

_BLOB = bytes([0xFF, 0xFE, 0x00, 0x01])


def _skill(*, user_id, name: str) -> Skill:
    return Skill(
        user_id=user_id,
        name=name,
        description=f"Do {name} when asked.",
        enabled=True,
        requires_mcp_servers=["eve_retrieval"],
        files=[
            SkillFile(path="scripts/roll.py", content=b"print(1)\n"),
            SkillFile(path="SKILL.md", content=b"---\nname: x\n---\n"),
            SkillFile(path="assets/blob.bin", content=_BLOB),
        ],
    )


@pytest.mark.asyncio
async def test_list_skills_requires_auth(async_client):
    response = await async_client.get("/skills")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_skills_returns_files_by_size_not_content(async_client):
    me, token = await create_test_user_and_token()
    other, _ = await create_test_user_and_token()
    mine = _skill(user_id=me.id, name="zz-api-mine")
    mine.enabled = False
    global_skill = _skill(user_id=None, name="zz-api-global")
    theirs = _skill(user_id=other.id, name="zz-api-other")
    deleted = _skill(user_id=me.id, name="zz-api-deleted")
    deleted.deleted_at = datetime.now(timezone.utc)
    rows = [mine, global_skill, theirs, deleted]
    for row in rows:
        await row.save()
    try:
        response = await async_client.get(
            "/skills?limit=100&page=1",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        body = response.json()
        names = {item["name"] for item in body["data"]}
        assert {"zz-api-mine", "zz-api-global"} <= names
        assert "zz-api-other" not in names
        assert "zz-api-deleted" not in names
        listed = next(item for item in body["data"] if item["name"] == "zz-api-mine")
        assert listed["enabled"] is False
        assert listed["requires_mcp_servers"] == ["eve_retrieval"]
        assert [item["path"] for item in listed["files"]] == [
            "SKILL.md",
            "assets/blob.bin",
            "scripts/roll.py",
        ]
        by_path = {item["path"]: item for item in listed["files"]}
        assert by_path["assets/blob.bin"]["size"] == len(_BLOB)
        assert by_path["scripts/roll.py"]["size"] == len(b"print(1)\n")
        assert set(by_path["SKILL.md"]) == {"path", "size"}
    finally:
        await cleanup_models([*rows, me, other])


@pytest.mark.asyncio
async def test_get_skill_hides_another_users_row(async_client):
    me, token = await create_test_user_and_token()
    other, _ = await create_test_user_and_token()
    theirs = _skill(user_id=other.id, name="zz-api-hidden")
    await theirs.save()
    try:
        missing = await async_client.get(
            f"/skills/{theirs.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert missing.status_code == 404
        own = _skill(user_id=me.id, name="zz-api-own")
        await own.save()
        found = await async_client.get(
            f"/skills/{own.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert found.status_code == 200
        body = found.json()
        blob = next(item for item in body["files"] if item["path"] == "assets/blob.bin")
        assert blob["size"] == len(_BLOB)
        assert set(blob) == {"path", "size"}
        await own.delete()
    finally:
        await cleanup_models([theirs, me, other])


@pytest.mark.asyncio
async def test_get_skill_rejects_a_bad_id(async_client):
    me, token = await create_test_user_and_token()
    try:
        response = await async_client.get(
            "/skills/not-an-id",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404
    finally:
        await cleanup_models([me])
