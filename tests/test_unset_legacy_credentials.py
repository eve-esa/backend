"""Tests for the src.commands.unset_legacy_credentials one-off sweep."""

import uuid
from datetime import datetime, timezone

import pytest

from src.commands.unset_legacy_credentials import (
    LEGACY_FIELDS,
    unset_legacy_credentials,
)
from src.database.models.user import User

HASH = "f" * 64


async def _insert_legacy_user() -> dict:
    """A user document as the pre-OIDC application left it."""
    doc = {
        "email": f"legacy-{uuid.uuid4().hex[:8]}@example.com",
        "password_hash": HASH,
        "activation_code": "AB12CD",
        "is_active": True,
        "first_name": "Legacy",
        "last_name": "User",
        "rate_limit_group": "eve_free",
        "rate_limit_tokens_used": 42,
        "timestamp": datetime.now(timezone.utc),
    }
    result = await User.get_collection().insert_one(doc)
    # Read back rather than reuse ``doc``: Mongo truncates the timestamp to
    # milliseconds, so this is what later reads compare against.
    return await User.get_collection().find_one({"_id": result.inserted_id})


async def _reload(doc: dict) -> dict:
    return await User.get_collection().find_one({"_id": doc["_id"]})


@pytest.mark.asyncio
async def test_dry_run_counts_and_writes_nothing(capsys):
    legacy = await _insert_legacy_user()
    try:
        summary = await unset_legacy_credentials()

        assert summary["password_hash"] >= 1
        assert summary["is_active"] >= 1
        assert summary["activation_code"] >= 1
        assert summary["documents"] >= 1
        assert "matched" not in summary
        assert await _reload(legacy) == legacy
        assert HASH not in capsys.readouterr().out
    finally:
        await User.get_collection().delete_one({"_id": legacy["_id"]})


@pytest.mark.asyncio
async def test_apply_unsets_only_the_legacy_fields(capsys):
    legacy = await _insert_legacy_user()
    try:
        summary = await unset_legacy_credentials(apply=True)

        assert summary["matched"] >= 1
        assert summary["modified"] == summary["matched"]
        stored = await _reload(legacy)
        for field in LEGACY_FIELDS:
            assert field not in stored
        expected = {k: v for k, v in legacy.items() if k not in LEGACY_FIELDS}
        assert stored == expected
        assert HASH not in capsys.readouterr().out
    finally:
        await User.get_collection().delete_one({"_id": legacy["_id"]})


@pytest.mark.asyncio
async def test_second_apply_is_a_no_op():
    legacy = await _insert_legacy_user()
    try:
        await unset_legacy_credentials(apply=True)
        after_first = await _reload(legacy)

        summary = await unset_legacy_credentials(apply=True)

        assert summary["documents"] == 0
        assert summary["matched"] == 0
        assert summary["modified"] == 0
        assert await _reload(legacy) == after_first
    finally:
        await User.get_collection().delete_one({"_id": legacy["_id"]})
