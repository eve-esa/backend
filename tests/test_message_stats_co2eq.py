import logging
from types import SimpleNamespace

import pytest

from src.routers import message as message_router


class _FakeCursor:
    async def to_list(self, length=None):
        return [
            {
                "message_count": 2,
                "input_characters": 400,
                "output_characters": 600,
                "total_characters": 1000,
            }
        ]


class _FakeMessages:
    def aggregate(self, _pipeline, allowDiskUse=False):
        return _FakeCursor()


@pytest.fixture
def stats_env(monkeypatch):
    async def find_all(filter_dict=None, **_kwargs):
        return [SimpleNamespace(id="c1")]

    monkeypatch.setattr(message_router.Conversation, "find_all", find_all)
    monkeypatch.setattr(
        message_router.Message, "get_collection", lambda: _FakeMessages()
    )
    monkeypatch.setattr(message_router, "_no_comparison_items_logged", False)


async def _call():
    return await message_router.get_my_message_stats(
        requesting_user=SimpleNamespace(id="u1")
    )


@pytest.mark.asyncio
async def test_empty_comparison_set_warns_once_without_stack(
    stats_env, monkeypatch, caplog
):
    async def no_rows(_usage_kg):
        return None

    monkeypatch.setattr(message_router, "closest_with_direction", no_rows)

    with caplog.at_level(logging.DEBUG, logger=message_router.logger.name):
        first = await _call()
        second = await _call()

    assert first["message_count"] == 2
    assert first["total_characters"] == 1000
    assert first["text"] == ""
    assert "co2eq_kg" in first
    assert second["text"] == ""

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is None
    assert "co2eq_comparisons" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_unexpected_equivalence_failure_stays_an_error(
    stats_env, monkeypatch, caplog
):
    async def boom(_usage_kg):
        raise ValueError("database unreachable")

    monkeypatch.setattr(message_router, "build_equivalence_sentence", boom)

    with caplog.at_level(logging.DEBUG, logger=message_router.logger.name):
        result = await _call()

    assert result["text"] == ""
    assert result["total_characters"] == 1000
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
