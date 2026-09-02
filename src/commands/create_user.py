"""Create the EVE row for a person who already exists at the identity provider.

Passwords are not this application's business any more, so this command no
longer sets one. What it does is pre-create the row a provider account will
attach to on first sign-in, which is what makes the local seed exercise the
legacy-link path rather than the provisioning path, and what gives
``seed_demo_artifacts`` a user to hang a conversation on.

    python -m src.commands.create_user <email> [first_name] [last_name]

Idempotent: an existing row with that address is left alone.
"""

import asyncio
import logging
import sys
from typing import Optional

from src.config import configure_logging
from src.database.models.user import User
from src.database.mongo import async_mongo_manager

configure_logging()
logger = logging.getLogger(__name__)


async def create_user(
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> str:
    """Create (or find) the user row for ``email`` and return its id."""
    await async_mongo_manager.connect()

    normalized = email.strip().lower()
    existing = await User.find_one({"email": normalized})
    if existing:
        logger.info("User %s already exists with id %s", normalized, existing.id)
        print(f"User {normalized} already exists (id {existing.id})")
        return existing.id

    user = await User.create(
        email=normalized,
        first_name=first_name,
        last_name=last_name,
    )
    logger.info("User %s created with id %s", normalized, user.id)
    print(f"User {normalized} created (id {user.id})")
    print("Sign in with this address at the identity provider; no password is stored here.")
    return user.id


if __name__ == "__main__":
    USAGE = "Usage: python -m src.commands.create_user <email> [first_name] [last_name]"

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or not args[0]:
        print(USAGE)
        sys.exit(1)

    asyncio.run(
        create_user(
            args[0],
            args[1] if len(args) > 1 else None,
            args[2] if len(args) > 2 else None,
        )
    )
