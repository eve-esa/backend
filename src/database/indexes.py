import logging

from src.database.mongo import get_collection

logger = logging.getLogger(__name__)


async def ensure_indexes() -> None:
    """Create MongoDB indexes required by MCP proxy, usage tracking, and API keys."""
    mcp_servers = get_collection("mcp_servers")
    mcp_usage = get_collection("mcp_usage")
    openai_usage = get_collection("openai_usage")
    api_keys = get_collection("api_keys")
    user_custom_models = get_collection("user_custom_models")
    catalog_platform_models = get_collection("catalog_platform_models")
    catalog_providers = get_collection("catalog_providers")

    await mcp_servers.create_index(
        [("user_id", 1), ("name", 1)],
        name="mcp_servers_user_name",
        unique=True,
    )

    # MCP proxy usage stats. Indexes mirror the back-office query dimensions:
    # per user, per server/tool, and per caller_type (login/api_key/internal),
    # always sliced by time. The bulky ``mcp_usage_payloads`` collection is keyed
    # by ``_id`` (auto-indexed) and needs no extra index.
    await mcp_usage.create_index(
        [("user_id", 1), ("timestamp", -1)],
        name="mcp_usage_by_user_time",
    )
    await mcp_usage.create_index(
        [("server_name", 1), ("tool_name", 1), ("timestamp", -1)],
        name="mcp_usage_by_server_tool_time",
    )
    await mcp_usage.create_index(
        [("caller_type", 1), ("timestamp", -1)],
        name="mcp_usage_by_caller_time",
    )
    await mcp_usage.create_index(
        [("api_key_id", 1), ("timestamp", -1)],
        name="mcp_usage_by_api_key_time",
    )

    # OpenAI proxy usage stats (previously unindexed). Same dimensions plus model.
    await openai_usage.create_index(
        [("user_id", 1), ("timestamp", -1)],
        name="openai_usage_by_user_time",
    )
    await openai_usage.create_index(
        [("model", 1), ("timestamp", -1)],
        name="openai_usage_by_model_time",
    )
    await openai_usage.create_index(
        [("caller_type", 1), ("timestamp", -1)],
        name="openai_usage_by_caller_time",
    )
    await openai_usage.create_index(
        [("api_key_id", 1), ("timestamp", -1)],
        name="openai_usage_by_api_key_time",
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

    await user_custom_models.create_index(
        [("user_id", 1), ("display_name", 1)],
        name="user_custom_models_user_display_name",
        unique=True,
        partialFilterExpression={"deleted_at": None},
    )
    await user_custom_models.create_index(
        [("user_id", 1), ("created_at", -1)],
        name="user_custom_models_user_created_at",
    )

    await catalog_platform_models.create_index(
        [("catalog_id", 1)],
        name="catalog_platform_models_catalog_id",
        unique=True,
    )
    await catalog_platform_models.create_index(
        [("enabled", 1), ("sort_order", 1)],
        name="catalog_platform_models_enabled_sort",
    )

    await catalog_providers.create_index(
        [("catalog_id", 1)],
        name="catalog_providers_catalog_id",
        unique=True,
    )
    await catalog_providers.create_index(
        [("enabled", 1), ("sort_order", 1)],
        name="catalog_providers_enabled_sort",
    )

    logger.info("MongoDB indexes ensured for MCP proxy features and API keys")
