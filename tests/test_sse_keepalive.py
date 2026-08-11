"""Unit tests for the SSE keepalive wrapper (src/utils/sse_keepalive.py)."""

import asyncio

import pytest

from src.utils.sse_keepalive import SSE_KEEPALIVE_COMMENT, with_sse_keepalive

pytestmark = [pytest.mark.asyncio, pytest.mark.no_db]


async def test_items_pass_through_unchanged():
    async def source():
        yield "a"
        yield "b"

    out = [item async for item in with_sse_keepalive(source(), interval=1.0)]
    assert out == ["a", "b"]


async def test_keepalive_comments_fill_silence():
    async def source():
        yield "first"
        await asyncio.sleep(0.25)
        yield "second"

    out = [item async for item in with_sse_keepalive(source(), interval=0.05)]
    assert out[0] == "first"
    assert out[-1] == "second"
    assert SSE_KEEPALIVE_COMMENT in out
    assert [i for i in out if i != SSE_KEEPALIVE_COMMENT] == ["first", "second"]


async def test_source_exception_propagates():
    async def source():
        yield "x"
        raise RuntimeError("boom")

    gen = with_sse_keepalive(source(), interval=1.0)
    assert await gen.__anext__() == "x"
    with pytest.raises(RuntimeError, match="boom"):
        await gen.__anext__()
    await gen.aclose()


async def test_early_close_cancels_pending_read():
    cancelled = asyncio.Event()

    async def source():
        yield "x"
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        yield "never"

    gen = with_sse_keepalive(source(), interval=0.05)
    assert await gen.__anext__() == "x"
    assert await gen.__anext__() == SSE_KEEPALIVE_COMMENT
    await gen.aclose()
    await asyncio.wait_for(cancelled.wait(), timeout=1)
