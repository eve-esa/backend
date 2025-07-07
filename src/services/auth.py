from jose import jwt
from datetime import datetime, timedelta
from uuid import uuid4
import hashlib
from src.database.models.user import User
from src.config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS,
)
import logging

logger = logging.getLogger(__name__)


async def verify_user(email: str, password: str) -> bool:
    user = await User.find_one({"email": email})
    if not user:
        return False

    return user.password_hash == hashlib.sha256(password.encode()).hexdigest()


def create_access_token(*, sub: str):
    expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expire, "sub": sub}
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(*, sub: str):
    expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    jti = str(uuid4())
    to_encode = {"exp": expire, "sub": sub, "jti": jti}
    token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token
