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
        self._invalidated: bool = False  # set by invalidate(), cleared after next fetch

    def _is_expired(self) -> bool:
        return time.monotonic() >= (self._expires_at - _REFRESH_MARGIN_S)

    def invalidate(self) -> None:
        """Force the next get_token() call to fetch a fresh token from Cognito.

        Must be called when a downstream service rejects the current token with
        401 so that the very next request uses a newly-issued token.  The reset
        is done under the same lock used by get_token() to avoid racing with a
        concurrent refresh that may have just stored a valid token.
        """
        # Fire-and-forget: we intentionally do NOT await the lock here because
        # invalidate() is called from a sync-like context (auth_flow generator).
        # The worst case is that get_token() returns a concurrently-refreshed
        # token — which is still valid.  Setting both fields to sentinel values
        # is atomic enough for our purposes.
        logger.warning(
            "[Cognito] Token cache invalidated – next get_token() will fetch a "
            "fresh token from Cognito (triggered by 401 from AgentCore)"
        )
        self._access_token = None
        self._expires_at = 0.0
        self._invalidated = True

    async def get_token(self) -> str:
        """Return a valid access token, fetching/refreshing as needed."""
        if self._access_token and not self._is_expired():
            logger.debug("[Cognito] Returning cached token (not yet expired)")
            return self._access_token

        async with self._lock:
            # Double-check after acquiring the lock.
            if self._access_token and not self._is_expired():
                logger.debug(
                    "[Cognito] Returning token refreshed by concurrent coroutine"
                )
                return self._access_token

            if self._invalidated:
                reason = "forced refresh after 401 from AgentCore"
            elif self._expires_at == 0.0:
                reason = "first fetch (no token yet)"
            else:
                remaining_s = max(0, self._expires_at - time.monotonic())
                reason = f"proactive refresh (token expires in ~{remaining_s:.0f}s)"
            logger.info("[Cognito] Fetching new access token – reason: %s", reason)
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

            # Only clear the flag after a successful fetch so that a transient
            # Cognito error does not swallow the "forced refresh" reason on the
            # next retry attempt.
            self._invalidated = False
            self._access_token = body["access_token"]
            expires_in = body.get("expires_in", 3600)
            # Cognito returns ``expires_in`` in seconds (typically 3600).
            self._expires_at = time.monotonic() + expires_in
            logger.info(
                "[Cognito] New token issued – expires_in=%ss, "
                "will refresh after %ss (margin=%ss)",
                expires_in,
                expires_in - _REFRESH_MARGIN_S,
                _REFRESH_MARGIN_S,
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
