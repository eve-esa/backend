"""Thumbs feedback on a message as Langfuse scores.

Traces reach Langfuse through the OTel collector; this module only posts
scores to ``POST /api/public/scores`` so a thumbs up or down lands on the trace
of the answer it rates. Off unless ``FEATURE_LANGFUSE_SCORES`` is true and
``LANGFUSE_HOST``, ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set.

Scores per message, each only when its field is set:

- ``thumbs-<message_id>``: ``feedback``, 1 for positive, 0 for negative,
  comment ``feedback_reason``.
- ``hallucination-<message_id>``: ``hallucination.feedback``, same values,
  comment ``hallucination.feedback_reason``.

The ids are deterministic, so Langfuse upserts: a changed vote replaces the old
score instead of adding one. A message without ``trace_id`` (telemetry was off
when it was generated) is skipped silently. The feedback is already in Mongo
before this runs; a Langfuse failure is logged and never reaches the caller.

Read them back with ``GET /api/public/v3/scores?traceId=...&fields=subject,details``:
Langfuse v4 answers 404 on the older ``GET /api/public/scores``.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set

import httpx

from src import config

logger = logging.getLogger(__name__)

SCORES_PATH = "/api/public/scores"
TIMEOUT_S = 3.0
THUMBS = "thumbs"
HALLUCINATION = "hallucination"

_VALUES = {"positive": 1, "negative": 0}
# Langfuse rejects environments outside this pattern with a 400.
_ENVIRONMENT_RE = re.compile(r"^(?!langfuse)[a-z0-9_-]{1,40}$")

# Strong references to in-flight posts: the loop only keeps weak ones.
_pending: Set["asyncio.Task[None]"] = set()
# Tests swap in an httpx.MockTransport; None is the real network.
_transport: Optional[httpx.AsyncBaseTransport] = None


def is_enabled() -> bool:
    return bool(
        config.FEATURE_LANGFUSE_SCORES
        and config.LANGFUSE_HOST
        and config.LANGFUSE_PUBLIC_KEY
        and config.LANGFUSE_SECRET_KEY
    )


def _environment() -> Optional[str]:
    from src.observability import deployment_environment

    value = deployment_environment().strip().lower()
    return value if _ENVIRONMENT_RE.match(value) else None


def _score(
    kind: str,
    message_id: str,
    trace_id: str,
    vote: Any,
    reason: Any,
    environment: Optional[str],
) -> Optional[Dict[str, Any]]:
    value = _VALUES.get(str(vote)) if vote is not None else None
    if value is None:
        return None
    score: Dict[str, Any] = {
        "id": f"{kind}-{message_id}",
        "traceId": trace_id,
        "name": kind,
        "dataType": "BOOLEAN",
        "value": value,
    }
    if reason:
        score["comment"] = str(reason)
    if environment:
        score["environment"] = environment
    return score


def build_scores(message: Any) -> List[Dict[str, Any]]:
    """Score bodies for ``message``; empty without ``trace_id`` or votes."""
    trace_id = getattr(message, "trace_id", None)
    if not trace_id:
        return []
    message_id = str(message.id)
    environment = _environment()
    hallucination = getattr(message, "hallucination", None) or {}
    candidates = [
        _score(
            THUMBS,
            message_id,
            trace_id,
            getattr(message, "feedback", None),
            getattr(message, "feedback_reason", None),
            environment,
        ),
        _score(
            HALLUCINATION,
            message_id,
            trace_id,
            hallucination.get("feedback"),
            hallucination.get("feedback_reason"),
            environment,
        ),
    ]
    return [score for score in candidates if score is not None]


async def post_scores(scores: List[Dict[str, Any]]) -> None:
    """Post each score; log and move on when one fails. Never raises."""
    try:
        async with httpx.AsyncClient(
            base_url=config.LANGFUSE_HOST,
            auth=(config.LANGFUSE_PUBLIC_KEY, config.LANGFUSE_SECRET_KEY),
            timeout=TIMEOUT_S,
            transport=_transport,
        ) as client:
            for score in scores:
                try:
                    response = await client.post(SCORES_PATH, json=score)
                    response.raise_for_status()
                except Exception as exc:
                    logger.warning(
                        "Langfuse score %s not sent: %s: %s",
                        score["id"],
                        type(exc).__name__,
                        exc,
                    )
    except Exception as exc:
        logger.warning("Langfuse scores not sent: %s: %s", type(exc).__name__, exc)


def schedule_feedback_scores(message: Any) -> Optional["asyncio.Task[None]"]:
    """Fire and forget the scores of ``message``. Never raises.

    Returns the task (tests await it) or None when nothing is sent.
    """
    try:
        if not is_enabled():
            return None
        scores = build_scores(message)
        if not scores:
            return None
        task = asyncio.get_running_loop().create_task(post_scores(scores))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
        return task
    except Exception as exc:
        logger.warning("Langfuse scores not scheduled: %s: %s", type(exc).__name__, exc)
        return None
