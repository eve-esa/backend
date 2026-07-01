import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.config import JWT_ALGORITHM, JWT_AUDIENCE_ACCESS, JWT_SECRET_KEY
from src.database.models.api_key import ApiKey
from src.database.models.user import User

security = HTTPBearer()
optional_bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)

_API_KEY_RE = re.compile(r"^eve_[0-9a-f]{64}$")

# caller_type values used by proxy usage tracking / back-office stats.
CALLER_TYPE_LOGIN = "login"          # human session, JWT access token
CALLER_TYPE_API_KEY = "api_key"      # programmatic ``eve_`` API key


@dataclass(frozen=True)
class Principal:
    """Authenticated caller identity resolved from a bearer token.

    Carries enough to attribute proxy usage in the back office: the stable
    ``user_id``, how they authenticated (``auth_type``), and — for API keys —
    which key (``api_key_id``) so usage can be broken down per key.
    """

    user_id: str
    auth_type: str  # "jwt" | "api_key"
    api_key_id: Optional[str] = None

    def caller_type(self) -> str:
        """Map this principal to a back-office ``caller_type`` dimension."""
        return CALLER_TYPE_API_KEY if self.auth_type == "api_key" else CALLER_TYPE_LOGIN


def verify_access_token(token: str) -> dict:
    """Decode and verify a user access JWT, returning its claims.

    Raises ``jose.JWTError`` on signature/audience/format failures.
    """
    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE_ACCESS,
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
    """Resolve a raw bearer token (JWT or ``eve_`` API key) to a :class:`Principal`.

    Used by ASGI proxy middleware that cannot use FastAPI ``Depends``.
    Raises ``PermissionError`` on any auth failure.
    """
    if token.startswith("eve_"):
        user_id, api_key_id = await _verify_api_key(token)
        return Principal(user_id=user_id, auth_type="api_key", api_key_id=api_key_id)
    try:
        claims = verify_access_token(token)
    except JWTError as exc:
        raise PermissionError("Invalid token") from exc
    user_id = claims.get("sub")
    if not user_id:
        raise PermissionError("Invalid token payload")
    return Principal(user_id=user_id, auth_type="jwt")


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Parse a raw ``Authorization`` header value into the credential token.

    Accepts JWT access tokens and ``eve_`` API keys. Returns ``None`` when the
    header is missing or not ``Bearer <token>``.
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
    """FastAPI dependency: required bearer credential (JWT or ``eve_`` API key)."""
    return credentials.credentials


async def get_optional_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer),
) -> Optional[str]:
    """FastAPI dependency: bearer credential when present, else ``None``."""
    if credentials is None:
        return None
    return credentials.credentials


async def get_user_id_from_bearer_token(token: str) -> str:
    """Resolve a raw bearer token (JWT or ``eve_`` API key) to a user_id string."""
    principal = await resolve_principal_from_bearer_token(token)
    return principal.user_id


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    token = credentials.credentials
    if token.startswith("eve_"):
        return await _get_user_from_api_key(token)
    try:
        payload = verify_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        user = await User.find_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
