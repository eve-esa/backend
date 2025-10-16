import logging
import sys
import asyncio
import secrets
import string
from typing import Optional
from src.services.utils import hash_password
from src.config import configure_logging

from src.database.models.user import User
from src.database.mongo import async_mongo_manager

configure_logging()
logger = logging.getLogger(__name__)


def generate_random_password(length: int = 20) -> str:
    """Generate a cryptographically secure random password of given length."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


async def create_user(email: str, password: Optional[str] = None) -> str:
    await async_mongo_manager.connect()
    if await User.find_one({"email": email}):
        print(f"Email {email} already exists")
        sys.exit(1)

    if not password:
        password = generate_random_password(20)

    password_hash = hash_password(password)
    user = await User.create(email=email, password_hash=password_hash)
    logger.info(
        f"User {email} created successfully with id {user.id} and password {password}"
    )
    # Output credentials as requested
    print(f"Email: {email}")
    print(f"Password: {password}")
    return password


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/commands/create_user.py <email>")
        sys.exit(1)

    email = sys.argv[1]

    if not email:
        print("Usage: python src/commands/create_user.py <email>")
        sys.exit(1)

    asyncio.run(create_user(email))
