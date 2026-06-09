import logging

from src.database.mongo import get_collection

logger = logging.getLogger(__name__)


async def ensure_indexes() -> None:
    """Create MongoDB indexes required by MCP proxy, usage tracking, and API keys."""
    mcp_servers = get_collection("mcp_servers")
    mcp_usage = get_collection("mcp_usage")
    api_keys = get_collection("api_keys")

    await mcp_servers.create_index(
        [("user_id", 1), ("name", 1)],
        name="mcp_servers_user_name",
        unique=True,
    )

    await mcp_usage.create_index(
        [("user_id", 1), ("timestamp", -1)],
        name="mcp_usage_by_user_time",
    )
    await mcp_usage.create_index(
        [("server_name", 1), ("timestamp", -1)],
        name="mcp_usage_by_server_time",
    )

    await api_keys.create_index(
        [("key_hash", 1)],
        name="api_keys_key_hash",
        unique=True,
    )
    await api_keys.create_index(
        [("user_id", 1)],
        name="api_keys_user_id",
    )

    logger.info("MongoDB indexes ensured for MCP proxy features and API keys")
