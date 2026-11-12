from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.services.generate_answer import (
    build_empty_answer_payload,
    persist_message_state,
)


def _message(metadata):
    return SimpleNamespace(
        output="", metadata=dict(metadata), stopped=False, save=AsyncMock()
    )


async def _persist(message, **kwargs):
    with patch(
        "src.services.generate_answer.Message.find_by_id_on_primary",
        AsyncMock(return_value=message),
    ):
        await persist_message_state("m1", **kwargs)


@pytest.mark.asyncio
async def test_a_successful_answer_drops_an_error_left_by_an_overlapping_retry():
    message = _message({"error": build_empty_answer_payload(), "pipeline": "agentic"})

    await _persist(message, output="answer", clear_error=True)

    assert message.output == "answer"
    assert message.metadata == {"pipeline": "agentic"}


@pytest.mark.asyncio
async def test_the_error_is_kept_without_clear_error():
    message = _message({"error": build_empty_answer_payload()})

    await _persist(message, output="partial")

    assert message.metadata["error"]["code"] == "empty_answer"


@pytest.mark.asyncio
async def test_a_new_error_wins_over_clear_error():
    message = _message({})
    error = {"code": "timeout", "type": "Timeout", "message": "late"}

    await _persist(message, error=error, clear_error=True)

    assert message.metadata["error"] == error
