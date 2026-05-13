import logging
from datetime import datetime, timezone
from typing import Optional

from src.database.mongo import get_collection

logger = logging.getLogger(__name__)


async def track_usage(
    user_id: str,
    path: str,
    method: str,
    model: Optional[str] = None,
    request_body: Optional[dict] = None,
    response_body=None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    status_code: Optional[int] = None,
) -> None:
    """Persist OpenAI proxy usage events without affecting request flow."""
    try:
        collection = get_collection("openai_usage")
        await collection.insert_one(
            {
                "user_id": user_id,
                "path": path,
                "method": method,
                "model": model,
                "request_body": request_body,
                "response_body": response_body,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "status_code": status_code,
                "timestamp": datetime.now(timezone.utc),
            }
        )
    except Exception as exc:
        logger.warning("Failed to track OpenAI proxy usage: %s", exc)
