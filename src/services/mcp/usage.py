import logging
from datetime import datetime, timezone
from typing import Optional

from src.database.mongo import get_collection

logger = logging.getLogger(__name__)


async def track_usage(
    user_id: str,
    server_name: str,
    request_method: str,
    status_code: Optional[int] = None,
) -> None:
    """Persist MCP proxy usage events without affecting request flow."""
    try:
        collection = get_collection("mcp_usage")
        await collection.insert_one(
            {
                "user_id": user_id,
                "server_name": server_name,
                "request_method": request_method,
                "status_code": status_code,
                "timestamp": datetime.now(timezone.utc),
            }
        )
    except Exception as exc:
        logger.warning("Failed to track MCP usage for server '%s': %s", server_name, exc)
