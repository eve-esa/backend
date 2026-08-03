"""Tests for the src.commands.register_dummy_mcp registration command."""

import pytest

from src.commands.register_dummy_mcp import (
    DUMMY_SERVER_NAME,
    DUMMY_SERVER_URL,
    register_dummy_mcp,
)
from src.database.models.mcp_server import MCPServer, ToolTransport
from tests.utils.cleaner import cleanup_models


@pytest.mark.asyncio
async def test_register_creates_new_server():
    """First call creates a new MCPServer record."""
    try:
        # Ensure clean state.
        await MCPServer.delete_many({"name": DUMMY_SERVER_NAME})

        await register_dummy_mcp(enabled=True)

        server = await MCPServer.find_one({"name": DUMMY_SERVER_NAME})
        assert server is not None
        assert server.name == DUMMY_SERVER_NAME
        assert server.description == "Local dummy MCP server (artifact e2e stand-in)"
        assert server.enabled is True
        assert server.config.url == DUMMY_SERVER_URL
        assert server.config.transport == ToolTransport.STREAMABLE_HTTP
        assert server.config.headers is None
    finally:
        await MCPServer.delete_many({"name": DUMMY_SERVER_NAME})


@pytest.mark.asyncio
async def test_register_idempotent_no_duplicate():
    """Re-running does not duplicate; updates the existing record."""
    try:
        await MCPServer.delete_many({"name": DUMMY_SERVER_NAME})

        # First call creates.
        await register_dummy_mcp(enabled=True)
        first_id = (await MCPServer.find_one({"name": DUMMY_SERVER_NAME})).id

        # Second call updates in place.
        await register_dummy_mcp(enabled=True)
        second_id = (await MCPServer.find_one({"name": DUMMY_SERVER_NAME})).id

        assert first_id == second_id, "Record should not be duplicated"

        # Verify count is exactly 1.
        count = await MCPServer.get_collection().count_documents(
            {"name": DUMMY_SERVER_NAME}
        )
        assert count == 1
    finally:
        await MCPServer.delete_many({"name": DUMMY_SERVER_NAME})


@pytest.mark.asyncio
async def test_register_disable_flag_toggles_enabled():
    """--disable flag sets enabled=False; default is enabled=True."""
    try:
        await MCPServer.delete_many({"name": DUMMY_SERVER_NAME})

        # Create as enabled.
        await register_dummy_mcp(enabled=True)
        server = await MCPServer.find_one({"name": DUMMY_SERVER_NAME})
        assert server.enabled is True

        # Update to disabled.
        await register_dummy_mcp(enabled=False)
        server = await MCPServer.find_one({"name": DUMMY_SERVER_NAME})
        assert server.enabled is False

        # Toggle back to enabled.
        await register_dummy_mcp(enabled=True)
        server = await MCPServer.find_one({"name": DUMMY_SERVER_NAME})
        assert server.enabled is True
    finally:
        await MCPServer.delete_many({"name": DUMMY_SERVER_NAME})


@pytest.mark.asyncio
async def test_register_preserves_other_fields_on_update():
    """Updated record keeps id and timestamps consistent."""
    try:
        await MCPServer.delete_many({"name": DUMMY_SERVER_NAME})

        await register_dummy_mcp(enabled=True)
        first = await MCPServer.find_one({"name": DUMMY_SERVER_NAME})
        first_created_at = first.created_at

        # Small delay to ensure timestamp would differ if it were reset.
        import asyncio
        await asyncio.sleep(0.01)

        await register_dummy_mcp(enabled=False)
        second = await MCPServer.find_one({"name": DUMMY_SERVER_NAME})

        assert second.id == first.id
        assert second.created_at == first_created_at
        assert second.updated_at >= first.updated_at
    finally:
        await MCPServer.delete_many({"name": DUMMY_SERVER_NAME})
