"""Which MCP servers receive the caller's EVE token as ``X-EVE-Token``.

The token is the user's own credential (an OIDC access token or an ``eve_``
key) and can manage the user's API keys, so it goes only to servers that call
back into EVE as the user. Today that is the retrieval server
(``POST /retrieve``). Interim rule until a backend-minted, audience-bound token
replaces it: a row opts in with ``forward_user_token: true``; a global row
named ``eve_retrieval`` or ``eve_retrieval_<env>`` counts as opted in while the
field is absent, so existing rows keep working. ``false`` always wins.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

RETRIEVAL_SERVER_NAMES = frozenset(
    {"eve_retrieval", "eve_retrieval_dev", "eve_retrieval_staging", "eve_retrieval_prod"}
)

# Server names already announced at INFO by this process.
_announced: set[str] = set()


def forwards_user_token(
    *, name: str, user_id: Optional[str], forward_user_token: Optional[bool]
) -> bool:
    """True when the server row may receive the caller's EVE token."""
    if forward_user_token is not None:
        return bool(forward_user_token)
    # Reserved-name fallback for global rows only: a user can register a row
    # with any name, including ``eve_retrieval``.
    return user_id is None and name in RETRIEVAL_SERVER_NAMES


def announce_user_token_forwarded(server_name: str) -> None:
    """Log once per process, at INFO, the name of a server that gets the token."""
    if server_name in _announced:
        return
    _announced.add(server_name)
    logger.info("mcp.user_token_forwarded server=%s", server_name)
