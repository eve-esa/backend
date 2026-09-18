"""``src.services.token_rate_limiter`` in isolation: the atomic counter and the window.

``test_openai_proxy_budget.py`` covers the same mechanics through the real
``/v1`` dispatcher; this file is about the service functions directly, where
it is cheap to run many concurrent calls and to patch ``User.save`` to prove
nothing here ever takes the full-document-rewrite path that used to lose
concurrent increments (see the docstring on ``_ensure_active_window``).
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId
from fastapi import HTTPException

from src.database.models.user import User
from src.services.token_rate_limiter import (
    TokenBudgetExceeded,
    check_token_budget,
    consume_tokens_for_user,
    enforce_token_budget_or_raise,
)
from tests.utils.cleaner import cleanup_models
from tests.utils.openai_proxy import seed_active_window
from tests.utils.utils import create_test_user_and_token

_PINNED_SETTINGS = {
    "enabled": True,
    "default_group": "eve_free",
    "groups": {"eve_free": {"max_tokens": 100, "period_months": 1}},
}


@pytest.fixture
def pinned_budget(monkeypatch):
    monkeypatch.setattr(
        "src.services.token_rate_limiter._rate_limit_settings", lambda: _PINNED_SETTINGS
    )
    return _PINNED_SETTINGS


# ── Atomic consume ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_consume_from_stale_instances_ends_at_exact_total(pinned_budget):
    """10 stale ``User`` copies, all loaded before any of them writes, each charge 7.

    A ``save()``-based consume would have each stale copy overwrite the
    others' increments with its own in-memory (zero) starting point; the
    atomic ``$inc`` cannot lose one no matter the interleaving.
    """
    user, _ = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=0)
        stale_instances = [await User.find_by_id(user.id) for _ in range(10)]

        await asyncio.gather(*(consume_tokens_for_user(u, 7) for u in stale_instances))

        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 70
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_consume_does_not_clobber_a_concurrent_private_document_inc(pinned_budget):
    """A concurrent, unrelated ``$inc`` on the same document must survive.

    This is the field ``user.save()`` used to overwrite on every proxied
    request (see ``src/services/private_document_limit.py``): the reproduction
    for finding 0.1 in the design doc.
    """
    user, _ = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=0)
        stale = await User.find_by_id(user.id)

        await asyncio.gather(
            consume_tokens_for_user(stale, 10),
            User.get_collection().update_one(
                {"_id": ObjectId(user.id)}, {"$inc": {"private_document_count": 3}}
            ),
        )

        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 10
        assert refreshed.private_document_count == 3
    finally:
        await cleanup_models([user])


# ── Rollover ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rollover_happens_once_under_concurrency(pinned_budget):
    """Several requests all see the same expired window; only one may roll it over.

    The compare-and-set is keyed on the exact ``rate_limit_period_end`` each
    caller loaded, so every loser reloads what the winner wrote instead of
    stomping it. Five concurrent charges of 4 after a rollover from a stale
    999 must land at exactly 20, not 999 + 20 and not less than 20.
    """
    user, _ = await create_test_user_and_token()
    try:
        past_end = datetime.now(timezone.utc) - timedelta(days=1)
        await User.get_collection().update_one(
            {"_id": ObjectId(user.id)},
            {
                "$set": {
                    "rate_limit_period_start": past_end - timedelta(days=30),
                    "rate_limit_period_end": past_end,
                    "rate_limit_tokens_used": 999,
                }
            },
        )
        stale_instances = [await User.find_by_id(user.id) for _ in range(5)]

        await asyncio.gather(*(consume_tokens_for_user(u, 4) for u in stale_instances))

        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 20
        # The Motor client is not tz_aware, so a raw find_by_id (unlike
        # _ensure_active_window's own in-memory normalisation) reads this
        # back naive; compare on equal footing rather than assume UTC twice.
        new_end = refreshed.rate_limit_period_end
        if new_end.tzinfo is None:
            new_end = new_end.replace(tzinfo=timezone.utc)
        assert new_end > past_end
    finally:
        await cleanup_models([user])


# ── check_token_budget / TokenBudgetExceeded ────────────────────────────────────


@pytest.mark.no_db
def test_token_budget_exceeded_message_and_retry_after():
    reset_at = datetime.now(timezone.utc) + timedelta(seconds=30)
    exceeded = TokenBudgetExceeded(group="eve_free", max_tokens=100, used_tokens=100, reset_at=reset_at)

    assert exceeded.message.startswith(
        "Token budget exceeded for group 'eve_free'. Limit is 100 tokens per period."
    )
    assert "Resets at" in exceeded.message
    retry_after = exceeded.retry_after_seconds()
    assert retry_after is not None and 1 <= retry_after <= 30


@pytest.mark.no_db
def test_token_budget_exceeded_without_reset_at_omits_retry_after():
    exceeded = TokenBudgetExceeded(group="eve_free", max_tokens=100, used_tokens=100, reset_at=None)

    assert "Resets at" not in exceeded.message
    assert exceeded.retry_after_seconds() is None


@pytest.mark.asyncio
async def test_check_token_budget_returns_none_under_the_cap(pinned_budget):
    user, _ = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=50)
        assert await check_token_budget(user) is None
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_check_token_budget_reports_group_and_used_tokens(pinned_budget):
    user, _ = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=100)
        exceeded = await check_token_budget(user)
        assert exceeded is not None
        assert exceeded.group == "eve_free"
        assert exceeded.max_tokens == 100
        assert exceeded.used_tokens == 100
    finally:
        await cleanup_models([user])


# ── enforce_token_budget_or_raise: unchanged text, added Retry-After ───────────


@pytest.mark.asyncio
async def test_enforce_keeps_detail_text_and_adds_retry_after(pinned_budget):
    user, _ = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=100)

        with pytest.raises(HTTPException) as exc_info:
            await enforce_token_budget_or_raise(user)

        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.detail.startswith("Token budget exceeded for group 'eve_free'.")
        assert exc.headers is not None
        assert int(exc.headers["Retry-After"]) >= 1
    finally:
        await cleanup_models([user])


@pytest.mark.asyncio
async def test_enforce_does_not_raise_under_the_cap(pinned_budget):
    user, _ = await create_test_user_and_token()
    try:
        await seed_active_window(user, used_tokens=0)
        await enforce_token_budget_or_raise(user)  # must not raise
    finally:
        await cleanup_models([user])


# ── No save() on the hot path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_save_on_the_hot_path(monkeypatch, pinned_budget):
    """``User.save`` patched to raise: both a fresh window and a charge still work.

    A first-ever call to ``_ensure_active_window`` (no window yet) is the one
    case that must still write, and it does so with the ``update_one``
    compare-and-set, never ``save()``.
    """

    user, _ = await create_test_user_and_token()
    try:
        async def _raise_save(self):
            raise AssertionError("save() must not be called on the hot path")

        # Patched only after the user is created: create_test_user_and_token
        # itself goes through User.create -> save(), which is not the code
        # under test here.
        monkeypatch.setattr(User, "save", _raise_save)

        await enforce_token_budget_or_raise(user)  # fresh window, under budget
        await consume_tokens_for_user(user, 5)

        refreshed = await User.find_by_id(user.id)
        assert refreshed.rate_limit_tokens_used == 5
    finally:
        await cleanup_models([user])
