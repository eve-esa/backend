"""OpenAI proxy usage tracking for back-office stats over *direct* proxy usage.

Design notes (collection layout)
--------------------------------
Two collections are written per event:

* ``openai_usage`` — one lean, heavily-indexed document per request: caller
  identity, model, token counts, status and latency. This is what the back
  office aggregates.
* ``openai_usage_payloads`` — the bulky request/response bodies, stored 1:1 by
  the same ``_id`` and fetched only when drilling into a single call.

Keeping the frequently-queried stats fields separate from the bulky bodies is
the MongoDB *Subset Pattern* / *Attribute split*: it keeps the hot collection's
documents (and indexes) small so token/usage aggregations scan far less data
than if every chat-completion body were inline.
  - Subset Pattern: https://www.mongodb.com/docs/manual/data-modeling/design-patterns/group-data/subset-pattern/
  - Data modeling intro: https://www.mongodb.com/docs/manual/data-modeling/

OpenAI and MCP deliberately use *separate* collections (not one polymorphic
collection) because the back office runs distinct stat sets per proxy and the
documents share almost no type-specific fields.
  - Polymorphic vs separate collections: https://www.mongodb.com/docs/manual/data-modeling/design-patterns/polymorphic-data/handle-different-document-types/
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.database.mongo import get_collection

logger = logging.getLogger(__name__)


async def track_usage(
    *,
    user_id: str,
    caller_type: str,
    path: str,
    method: str,
    api_key_id: Optional[str] = None,
    model: Optional[str] = None,
    streaming: bool = False,
    request_body: Optional[dict] = None,
    response_body=None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    status_code: Optional[int] = None,
    outcome: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> None:
    """Persist one OpenAI proxy usage event without affecting request flow.

    The lean event goes to ``openai_usage``; ``request_body``/``response_body``
    are offloaded to ``openai_usage_payloads`` under the same ``_id``.
    """
    try:
        collection = get_collection("openai_usage")
        result = await collection.insert_one(
            {
                "user_id": user_id,
                "caller_type": caller_type,
                "api_key_id": api_key_id,
                "path": path,
                "method": method,
                "model": model,
                "streaming": streaming,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "status_code": status_code,
                "outcome": outcome,
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        if request_body is not None or response_body is not None:
            payloads = get_collection("openai_usage_payloads")
            await payloads.insert_one(
                {
                    "_id": result.inserted_id,
                    "request_body": request_body,
                    "response_body": response_body,
                }
            )
    except Exception as exc:
        logger.warning("Failed to track OpenAI proxy usage: %s", exc)
