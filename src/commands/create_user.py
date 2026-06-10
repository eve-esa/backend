import logging
import sys
import asyncio
from typing import Optional
from src.config import configure_logging

from src.database.mongo import async_mongo_manager
from src.services.auth import create_user_admin

configure_logging()
logger = logging.getLogger(__name__)


async def create_user(email: str, password: Optional[str] = None) -> str:
    await async_mongo_manager.connect()
    try:
        user, plaintext_password = await create_user_admin(
            email=email,
            password=password,
            is_active=False,
        )
    except ValueError:
        print(f"Email {email} already exists")
        sys.exit(1)

    logger.info(
        f"User {email} created successfully with id {user.id} and password {plaintext_password}"
    )
    print(f"Email: {email}")
    print(f"Password: {plaintext_password}")
    return plaintext_password


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/commands/create_user.py <email>")
        sys.exit(1)

    email = sys.argv[1]

    if not email:
        print("Usage: python src/commands/create_user.py <email>")
        sys.exit(1)

    asyncio.run(create_user(email))
