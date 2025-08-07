import uuid
from typing import Optional, Tuple

from src.database.models.user import User
from src.database.mongo import async_mongo_manager
from src.services.utils import hash_password
from src.services.auth import create_access_token


async def create_test_user_and_token(
    *,
    email: Optional[str] = None,
    password: str = "password",
    first_name: str = "Test",
    last_name: str = "User",
) -> Tuple[User, str]:
    """Utility for tests that returns a freshly created user and a valid JWT.

    The user is persisted in the (mocked) MongoDB instance so that subsequent
    API calls executed by FastAPI can fetch it via the regular data-access
    helpers (``User.find_by_id`` et al.).
    """

    email = email or f"{uuid.uuid4().hex[:8]}@example.com"

    if async_mongo_manager.database is None:
        await async_mongo_manager.connect()

    existing = await User.find_one({"email": email})
    # this should never happen, but just in case
    if existing:
        await existing.delete()

    user = await User.create(
        email=email,
        password_hash=hash_password(password),
        first_name=first_name,
        last_name=last_name,
    )

    access_token: str = create_access_token(sub=user.id)
    return user, access_token
