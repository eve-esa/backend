import asyncio
import logging

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import OperationFailure

from src.database.mongo import get_collection

logger = logging.getLogger(__name__)

# DocumentDB allows one index build per collection at a time and rejects the rest with
# this code, unlike MongoDB which queues them. Gunicorn boots every worker at once, so
# on a fresh database the workers race and all but one lose. Losing is not an error: the
# winner is building the very index we want, so wait for it instead of failing startup.
# Staging first boot, 2026-07-30: "Existing index build in progress on the same
# collection" killed a worker, and gunicorn turns one failed worker into a dead task.
_CONCURRENT_INDEX_BUILD = 40333
_MAX_ATTEMPTS = 10
_BACKOFF_SECONDS = 0.5


async def _create_index(collection: AsyncIOMotorCollection, keys, **kwargs) -> None:
    """create_index tolerating another worker building the same index concurrently."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            await collection.create_index(keys, **kwargs)
            return
        except OperationFailure as exc:
            if exc.code != _CONCURRENT_INDEX_BUILD or attempt == _MAX_ATTEMPTS - 1:
                raise
            logger.info(
                "Index %s is being built by another worker, retrying (%d/%d)",
                kwargs.get("name", keys),
                attempt + 1,
                _MAX_ATTEMPTS,
            )
            await asyncio.sleep(_BACKOFF_SECONDS * (2**attempt))


async def ensure_indexes() -> None:
    """Create MongoDB indexes required by MCP proxy, usage tracking, and API keys."""
    mcp_servers = get_collection("mcp_servers")
    mcp_usage = get_collection("mcp_usage")
    openai_usage = get_collection("openai_usage")
    api_keys = get_collection("api_keys")
    user_custom_models = get_collection("user_custom_models")
    catalog_platform_models = get_collection("catalog_platform_models")
    catalog_providers = get_collection("catalog_providers")

    await _create_index(
        mcp_servers,
        [("user_id", 1), ("name", 1)],
        name="mcp_servers_user_name",
        unique=True,
    )

    # MCP proxy usage stats. Indexes mirror the back-office query dimensions:
    # per user, per server/tool, and per caller_type (login/api_key/internal),
    # always sliced by time. The bulky ``mcp_usage_payloads`` collection is keyed
    # by ``_id`` (auto-indexed) and needs no extra index.
    await _create_index(
        mcp_usage,
        [("user_id", 1), ("timestamp", -1)],
        name="mcp_usage_by_user_time",
    )
    await _create_index(
        mcp_usage,
        [("server_name", 1), ("tool_name", 1), ("timestamp", -1)],
        name="mcp_usage_by_server_tool_time",
    )
    await _create_index(
        mcp_usage,
        [("caller_type", 1), ("timestamp", -1)],
        name="mcp_usage_by_caller_time",
    )
    await _create_index(
        mcp_usage,
        [("api_key_id", 1), ("timestamp", -1)],
        name="mcp_usage_by_api_key_time",
    )

    # OpenAI proxy usage stats (previously unindexed). Same dimensions plus model.
    await _create_index(
        openai_usage,
        [("user_id", 1), ("timestamp", -1)],
        name="openai_usage_by_user_time",
    )
    await _create_index(
        openai_usage,
        [("model", 1), ("timestamp", -1)],
        name="openai_usage_by_model_time",
    )
    await _create_index(
        openai_usage,
        [("caller_type", 1), ("timestamp", -1)],
        name="openai_usage_by_caller_time",
    )
    await _create_index(
        openai_usage,
        [("api_key_id", 1), ("timestamp", -1)],
        name="openai_usage_by_api_key_time",
    )

    await _create_index(
        api_keys,
        [("key_hash", 1)],
        name="api_keys_key_hash",
        unique=True,
    )
    await _create_index(
        api_keys,
        [("user_id", 1)],
        name="api_keys_user_id",
    )

    await _create_index(
        user_custom_models,
        [("user_id", 1), ("display_name", 1)],
        name="user_custom_models_user_display_name",
        unique=True,
        partialFilterExpression={"deleted_at": None},
    )
    await _create_index(
        user_custom_models,
        [("user_id", 1), ("created_at", -1)],
        name="user_custom_models_user_created_at",
    )

    await _create_index(
        catalog_platform_models,
        [("catalog_id", 1)],
        name="catalog_platform_models_catalog_id",
        unique=True,
    )
    await _create_index(
        catalog_platform_models,
        [("enabled", 1), ("sort_order", 1)],
        name="catalog_platform_models_enabled_sort",
    )

    await _create_index(
        catalog_providers,
        [("catalog_id", 1)],
        name="catalog_providers_catalog_id",
        unique=True,
    )
    await _create_index(
        catalog_providers,
        [("enabled", 1), ("sort_order", 1)],
        name="catalog_providers_enabled_sort",
    )

    logger.info("MongoDB indexes ensured for MCP proxy features and API keys")
