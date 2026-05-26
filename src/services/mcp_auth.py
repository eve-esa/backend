"""AWS Cognito client-credentials token provider for AgentCore MCP servers.

Fetches and caches an OAuth2 access token using the client_credentials grant.
The cached token is refreshed automatically when it approaches expiry.

Reference:
    https://docs.aws.amazon.com/cognito/latest/developerguide/token-endpoint.html
"""

import asyncio
import logging
import time
from typing import Optional

import httpx

from src.config import (
    AGENTCORE_CLIENT_ID,
    AGENTCORE_CLIENT_SECRET,
    AGENTCORE_TOKEN_URL,
)

logger = logging.getLogger(__name__)

_REFRESH_MARGIN_S = 300  # refresh 5 minutes before expiry


class CognitoTokenProvider:
    """Thread-safe, async-safe token provider with automatic refresh.

    Tokens are cached in memory and refreshed when ``current_time >= expires_at -
    _REFRESH_MARGIN_S``.  A lock prevents concurrent token fetches (thundering-herd).
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
    ) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    def _is_expired(self) -> bool:
        return time.monotonic() >= (self._expires_at - _REFRESH_MARGIN_S)

    async def get_token(self) -> str:
        """Return a valid access token, fetching/refreshing as needed."""
        if self._access_token and not self._is_expired():
            return self._access_token

        async with self._lock:
            # Double-check after acquiring the lock.
            if self._access_token and not self._is_expired():
                return self._access_token

            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    self._token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=15.0,
                )
                resp.raise_for_status()
                body = resp.json()

            self._access_token = body["access_token"]
            # Cognito returns ``expires_in`` in seconds (typically 3600).
            self._expires_at = time.monotonic() + body.get("expires_in", 3600)
            logger.info(
                "Cognito token refreshed (expires_in=%ss)", body.get("expires_in")
            )
            return self._access_token


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_provider: Optional[CognitoTokenProvider] = None


def get_cognito_token_provider() -> Optional[CognitoTokenProvider]:
    """Return the shared :class:`CognitoTokenProvider`, or ``None`` if not configured."""
    global _provider
    if _provider is not None:
        return _provider
    if not (AGENTCORE_TOKEN_URL and AGENTCORE_CLIENT_ID and AGENTCORE_CLIENT_SECRET):
        logger.debug("AgentCore Cognito credentials not configured – MCP auth disabled")
        return None
    _provider = CognitoTokenProvider(
        token_url=AGENTCORE_TOKEN_URL,
        client_id=AGENTCORE_CLIENT_ID,
        client_secret=AGENTCORE_CLIENT_SECRET,
    )
    return _provider
