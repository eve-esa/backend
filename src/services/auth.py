from jose import jwt
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from src.services.utils import hash_password
from src.database.models.user import User
from src.config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS,
    JWT_AUDIENCE_ACCESS,
    JWT_AUDIENCE_REFRESH,
)
import logging

logger = logging.getLogger(__name__)


async def verify_user(email: str, password: str) -> bool:
    user = await User.find_one({"email": email})
    if not user:
        return False

    return user.password_hash == hash_password(password)


def create_access_token(*, sub: str):
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode = {"exp": expire, "sub": sub, "aud": JWT_AUDIENCE_ACCESS}
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(*, sub: str):
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    jti = str(uuid4())
    to_encode = {"exp": expire, "sub": sub, "jti": jti, "aud": JWT_AUDIENCE_REFRESH}
    token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token
