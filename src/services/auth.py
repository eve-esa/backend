import hashlib
import random
import secrets
import string
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from jose import jwt

from src.config import (
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_AUDIENCE_ACCESS,
    JWT_AUDIENCE_REFRESH,
    JWT_REFRESH_TOKEN_EXPIRE_DAYS,
    JWT_SECRET_KEY,
)
from src.database.models.user import User
from src.services.utils import hash_password


async def verify_user(email: str, password: str) -> bool:
    user = await User.find_one({"email": email})
    if not user:
        return False

    return user.password_hash == hash_password(password)


def generate_activation_code(length: int = 6) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


async def create_user(
    email: str,
    password: str,
    first_name: str | None = None,
    last_name: str | None = None,
) -> User:
    existing = await User.find_one({"email": email})
    if existing:
        raise ValueError("User with this email already exists")
    activation_code = generate_activation_code()
    user = await User.create(
        email=email,
        password_hash=hash_password(password),
        first_name=first_name,
        last_name=last_name,
        is_active=False,
        activation_code=activation_code,
    )
    return user


def create_access_token(*, sub: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expire, "sub": sub, "aud": JWT_AUDIENCE_ACCESS}
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(*, sub: str):
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    jti = str(uuid4())
    to_encode = {"exp": expire, "sub": sub, "jti": jti, "aud": JWT_AUDIENCE_REFRESH}
    token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def generate_api_key() -> tuple[str, str]:
    """Generate a new SHA-256 hash API key.

    Returns ``(raw_token, key_hash)`` where ``raw_token`` is shown to the user
    exactly once and ``key_hash`` is the SHA-256 digest stored in the DB.
    """
    raw = "eve_" + secrets.token_hex(32)
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, key_hash
