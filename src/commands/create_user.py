import hashlib
import logging
import sys
import os
import asyncio
from src.config import configure_logging

from src.database.models.user import User
from src.database.mongo import async_mongo_manager

configure_logging()
logger = logging.getLogger(__name__)

async def create_user(email: str, password: str):
    await async_mongo_manager.connect()
    user_id = await User.create(email=email, password_hash=password)
    logger.info(f"User {email} created successfully with id {user_id}")


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
