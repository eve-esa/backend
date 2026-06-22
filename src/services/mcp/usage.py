"""MCP proxy usage tracking for back-office stats over *direct* proxy usage.

Design notes (collection layout)
--------------------------------
Two collections are written per event:

* ``mcp_usage`` — one lean, heavily-indexed document per *logical* MCP
  operation (only ``tools/call`` is recorded; ``initialize``/``tools/list``/
  notifications are skipped upstream). This is what the back office aggregates.
* ``mcp_usage_payloads`` — the bulky part (tool-call arguments and the raw
  upstream response), stored 1:1 by the same ``_id`` and fetched only when
  drilling into a single call.

Keeping the frequently-queried stats fields separate from the rarely-read bulky
fields is the MongoDB *Subset Pattern* / *Attribute split*: it keeps the hot
collection's documents (and indexes) small so stats aggregations scan less data.
  - Subset Pattern: https://www.mongodb.com/docs/manual/data-modeling/design-patterns/group-data/subset-pattern/
  - Data modeling intro: https://www.mongodb.com/docs/manual/data-modeling/

OpenAI and MCP deliberately use *separate* collections (not one polymorphic
collection) because the back office runs distinct stat sets per proxy and the
documents share almost no type-specific fields; separating them avoids a sparse,
mixed-shape collection.
  - Polymorphic vs separate collections: https://www.mongodb.com/docs/manual/data-modeling/design-patterns/polymorphic-data/handle-different-document-types/
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.database.mongo import get_collection

logger = logging.getLogger(__name__)


async def track_usage(
    *,
    user_id: str,
    caller_type: str,
    server_name: str,
    operation: str,
    tool_name: Optional[str] = None,
    api_key_id: Optional[str] = None,
    status_code: Optional[int] = None,
    is_error: Optional[bool] = None,
    outcome: Optional[str] = None,
    latency_ms: Optional[float] = None,
    request_payload: Optional[Any] = None,
    response_payload: Optional[Any] = None,
) -> None:
    """Persist one MCP proxy usage event without affecting request flow.

    The lean event goes to ``mcp_usage``; ``request_payload``/``response_payload``
    (tool arguments and the raw upstream response) are offloaded to
    ``mcp_usage_payloads`` under the same ``_id``.
    """
    try:
        collection = get_collection("mcp_usage")
        result = await collection.insert_one(
            {
                "user_id": user_id,
                "caller_type": caller_type,
                "api_key_id": api_key_id,
                "server_name": server_name,
                "operation": operation,
                "tool_name": tool_name,
                "status_code": status_code,
                "is_error": is_error,
                "outcome": outcome,
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        if request_payload is not None or response_payload is not None:
            payloads = get_collection("mcp_usage_payloads")
            await payloads.insert_one(
                {
                    "_id": result.inserted_id,
                    "request_payload": request_payload,
                    "response_payload": response_payload,
                }
            )
    except Exception as exc:
        logger.warning("Failed to track MCP usage for server '%s': %s", server_name, exc)
