import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.database.models.api_key import ApiKey
from src.database.models.user import User
from src.services.identity import resolve_user_id
from src.services.oidc import IdentityProviderUnavailable, verify_access_token

security = HTTPBearer()
logger = logging.getLogger(__name__)

_API_KEY_RE = re.compile(r"^eve_[0-9a-f]{64}$")

# How the caller proved who they are. Two ways, and there is no third: an
# ``eve_`` API key issued by this application, or an access token from the
# identity provider. The application signs nothing itself any more.
AUTH_TYPE_API_KEY = "api_key"
AUTH_TYPE_OIDC = "oidc"

# caller_type values used by proxy usage tracking / back-office stats. This is a
# persisted analytics dimension, so it names what the caller IS (a human session
# or a machine key) rather than which protocol they arrived on: renaming "login"
# to "oidc" would zero an existing back-office series to say the same thing.
CALLER_TYPE_LOGIN = "login"          # human session, provider access token
CALLER_TYPE_API_KEY = "api_key"      # programmatic ``eve_`` API key


@dataclass(frozen=True)
class Principal:
    """Authenticated caller identity resolved from a bearer token.

    Carries enough to attribute proxy usage in the back office: the stable
    ``user_id``, how they authenticated (``auth_type``), and — for API keys —
    which key (``api_key_id``) so usage can be broken down per key.
    """

    user_id: str
    auth_type: str  # AUTH_TYPE_OIDC | AUTH_TYPE_API_KEY
    api_key_id: Optional[str] = None

    def caller_type(self) -> str:
        """Map this principal to a back-office ``caller_type`` dimension."""
        return (
            CALLER_TYPE_API_KEY
            if self.auth_type == AUTH_TYPE_API_KEY
            else CALLER_TYPE_LOGIN
        )


async def _verify_api_key(token: str) -> tuple[str, str]:
    """Validate an ``eve_`` API key, stamp ``last_used_at``, return ``(user_id, key_id)``.

    Raises ``PermissionError`` on any failure so callers can decide whether to
    surface it as an ``HTTPException`` or propagate it as a plain exception.
    """
    if not _API_KEY_RE.fullmatch(token):
        raise PermissionError("Invalid API key format")
    key_hash = hashlib.sha256(token.encode()).hexdigest()
    now = datetime.now(timezone.utc)
    doc = await ApiKey.get_collection().find_one_and_update(
        {
            "key_hash": key_hash,
            "revoked_at": None,
            "$or": [
                {"expires_at": None},
                {"expires_at": {"$gt": now}},
            ],
        },
        {"$set": {"last_used_at": now}},
        projection={"user_id": 1},
    )
    if doc is None:
        raise PermissionError("Invalid or revoked API key")
    return str(doc["user_id"]), str(doc["_id"])


async def _get_user_from_api_key(token: str) -> User:
    try:
        user_id, _ = await _verify_api_key(token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    user = await User.find_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def resolve_principal_from_bearer_token(token: str) -> Principal:
    """Resolve a raw bearer token to a :class:`Principal`.

    One path per credential kind and nothing else: an ``eve_`` prefix is an API
    key, anything else is an identity-provider access token.

    Used by ASGI proxy middleware that cannot use FastAPI ``Depends``.
    Raises ``PermissionError`` on any auth failure.
    """
    if token.startswith("eve_"):
        user_id, api_key_id = await _verify_api_key(token)
        return Principal(
            user_id=user_id, auth_type=AUTH_TYPE_API_KEY, api_key_id=api_key_id
        )

    claims = await verify_access_token(token)
    user_id = await resolve_user_id(claims, token)
    return Principal(user_id=user_id, auth_type=AUTH_TYPE_OIDC)


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Parse a raw ``Authorization`` header value into the credential token.

    Accepts provider access tokens and ``eve_`` API keys. Returns ``None`` when
    the header is missing or not ``Bearer <token>``.
    """
    if not authorization_header:
        return None
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


async def get_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """FastAPI dependency: required bearer credential (access token or API key)."""
    return credentials.credentials


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    token = credentials.credentials
    if token.startswith("eve_"):
        return await _get_user_from_api_key(token)

    # The resolver raises PermissionError, which FastAPI has no handler for: left
    # unhandled every rejected token would answer 500 instead of 401.
    try:
        principal = await resolve_principal_from_bearer_token(token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except IdentityProviderUnavailable as exc:
        # The provider is down, the credential is not wrong. A 401 here would
        # send every signed-in user through a sign-in that cannot complete.
        logger.warning("Identity provider unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Identity provider unavailable")

    user = await User.find_by_id(principal.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
