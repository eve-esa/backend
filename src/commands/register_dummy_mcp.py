"""Register or update a dummy local MCP server for artifact e2e testing.

The dummy server serves as a stand-in for artifact e2e tests without requiring
a real MCP implementation. This command idempotently upserts an MCPServer record:

    python -m src.commands.register_dummy_mcp [--disable]

By default, the server is registered as enabled=True. Pass --disable to
set enabled=False (useful for temporary toggles).
"""

import argparse
import asyncio
import logging

from src.config import configure_logging
from src.database.models.mcp_server import MCPServer, ToolConfig, ToolTransport
from src.database.mongo import async_mongo_manager

configure_logging()
logger = logging.getLogger(__name__)

DUMMY_SERVER_NAME = "dummy"
DUMMY_SERVER_URL = "http://dummy-mcp:8000/mcp"
DUMMY_SERVER_DESCRIPTION = "Local dummy MCP server (artifact e2e stand-in)"


async def register_dummy_mcp(enabled: bool = True) -> None:
    """Register or update a dummy MCP server for testing.

    Upserts an MCPServer record with a fixed name; re-running updates enabled
    status in place without duplicating the record.

    Args:
        enabled (bool): Whether the server should be enabled (default: True).
    """
    # Reuse an already-open connection (e.g. the test suite's isolated test
    # database) instead of unconditionally reconnecting to the default URI.
    if async_mongo_manager.database is None:
        await async_mongo_manager.connect()

    existing = await MCPServer.find_one({"name": DUMMY_SERVER_NAME})

    config = ToolConfig(
        transport=ToolTransport.STREAMABLE_HTTP,
        url=DUMMY_SERVER_URL,
        headers=None,
    )

    if existing:
        # Update in place: toggle enabled status and refresh timestamp.
        existing.enabled = enabled
        existing.description = DUMMY_SERVER_DESCRIPTION
        existing.config = config
        await existing.save()
        status = "enabled" if enabled else "disabled"
        print(f"Updated existing MCP server '{DUMMY_SERVER_NAME}' (now {status})")
        logger.info(
            f"Dummy MCP server '{DUMMY_SERVER_NAME}' updated with enabled={enabled}"
        )
    else:
        # Create new record.
        server = await MCPServer.create(
            name=DUMMY_SERVER_NAME,
            description=DUMMY_SERVER_DESCRIPTION,
            enabled=enabled,
            config=config,
        )
        status = "enabled" if enabled else "disabled"
        print(f"Registered new MCP server '{DUMMY_SERVER_NAME}' (id={server.id}, {status})")
        logger.info(
            f"Dummy MCP server '{DUMMY_SERVER_NAME}' created with id={server.id}, "
            f"enabled={enabled}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register or update a dummy local MCP server for artifact e2e testing."
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Register the server as disabled (enabled=False). Default is enabled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    enabled = not args.disable
    asyncio.run(register_dummy_mcp(enabled=enabled))
