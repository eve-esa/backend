import hashlib
import logging
from datetime import datetime, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.config import JWT_ALGORITHM, JWT_AUDIENCE_ACCESS, JWT_SECRET_KEY
from src.database.models.api_key import ApiKey
from src.database.models.user import User

security = HTTPBearer()
logger = logging.getLogger(__name__)


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


async def _get_user_from_api_key(token: str) -> User:
    key_hash = hashlib.sha256(token.encode()).hexdigest()
    api_key = await ApiKey.find_one({"key_hash": key_hash})
    if not api_key or not api_key.is_valid:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    user = await User.find_by_id(api_key.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    api_key.last_used_at = datetime.now(timezone.utc)
    await api_key.save()
    return user


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
