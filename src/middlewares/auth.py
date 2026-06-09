import hashlib
import logging
import re
from datetime import datetime, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.config import JWT_ALGORITHM, JWT_AUDIENCE_ACCESS, JWT_SECRET_KEY
from src.database.models.api_key import ApiKey
from src.database.models.user import User

security = HTTPBearer()
logger = logging.getLogger(__name__)

_API_KEY_RE = re.compile(r"^eve_[0-9a-f]{64}$")


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


async def _verify_api_key(token: str) -> str:
    """Validate an ``eve_`` API key, stamp ``last_used_at``, and return its ``user_id``.

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
    return str(doc["user_id"])


async def _get_user_from_api_key(token: str) -> User:
    try:
        user_id = await _verify_api_key(token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    user = await User.find_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_user_id_from_bearer_token(token: str) -> str:
    """Resolve a raw bearer token (JWT or ``eve_`` API key) to a user_id string.

    Used by ASGI proxy middleware that cannot use FastAPI ``Depends``.
    Raises ``PermissionError`` on any auth failure.
    """
    if token.startswith("eve_"):
        return await _verify_api_key(token)
    try:
        claims = verify_access_token(token)
    except JWTError as exc:
        raise PermissionError("Invalid token") from exc
    user_id = claims.get("sub")
    if not user_id:
        raise PermissionError("Invalid token payload")
    return user_id


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
