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


async def create_user(
    email: str, password: Optional[str] = None, activate: bool = False
) -> str:
    await async_mongo_manager.connect()
    if await User.find_one({"email": email}):
        print(f"Email {email} already exists")
        sys.exit(1)

    if not password:
        password = generate_random_password(20)

    password_hash = hash_password(password)
    user = await User.create(email=email, password_hash=password_hash)
    if activate:
        # Mirror the activation flow in src/routers/auth.py so seeded e2e users
        # can log in without a verification round-trip.
        user.is_active = True
        user.activation_code = None
        await user.save()
    logger.info(
        f"User {email} created successfully with id {user.id} (active={user.is_active})"
    )
    # Output credentials as requested (operator-facing; never logged).
    print(f"Email: {email}")
    print(f"Password: {password}")
    return password


if __name__ == "__main__":
    USAGE = "Usage: python -m src.commands.create_user <email> [password] [--test]"

    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    positional = [a for a in args if not a.startswith("--")]

    if not positional or not positional[0]:
        print(USAGE)
        sys.exit(1)

    email = positional[0]
    password = positional[1] if len(positional) > 1 else None
    activate = bool(flags & {"--test", "--activate"})

    # The frontend login enforces a minimum password length of 8; keep the CLI
    # consistent so seeded users can actually authenticate.
    if password is not None and len(password) < 8:
        print("Error: password must be at least 8 characters long")
        sys.exit(1)

    asyncio.run(create_user(email, password, activate))
