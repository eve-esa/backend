import logging
import sys
import asyncio
from src.services.utils import hash_password
from src.config import configure_logging

from src.database.models.user import User
from src.database.mongo import async_mongo_manager

configure_logging()
logger = logging.getLogger(__name__)


async def create_user(email: str, password: str):
    await async_mongo_manager.connect()
    if await User.find_one({"email": email}):
        print(f"Email {email} already exists")
        sys.exit(1)

    password_hash = hash_password(password)
    user = await User.create(email=email, password_hash=password_hash)
    logger.info(
        f"User {email} created successfully with id {user.id} and password {password}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/commands/create_user.py <email> <password>")
        sys.exit(1)

    email = sys.argv[1]
    password = sys.argv[2]

    if not email or not password:
        print("Usage: python src/commands/create_user.py <email> <password>")
        sys.exit(1)

    asyncio.run(create_user(email, password))
