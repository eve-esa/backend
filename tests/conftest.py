# tests/conftest.py
import os
import sys
from typing import AsyncIterator

import pytest
from mongomock_motor import AsyncMongoMockClient

# Make sure the project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Signal to the application code that we want to use the mock.
os.environ["USE_MOCK_MONGO"] = "1"

from src.database.mongo import (
    async_mongo_manager,
)


@pytest.fixture(scope="session", autouse=True)
async def _mongo_mock() -> AsyncIterator[None]:
    """Initialise a *single* in-memory MongoDB instance for the session."""
    client: AsyncMongoMockClient = AsyncMongoMockClient()
    db_name = os.getenv("MONGO_DATABASE", "eve_backend_test")

    # Wire the mocked client/database into the global manager used by the app.
    async_mongo_manager.client = client
    async_mongo_manager.database = client[db_name]

    try:
        yield
    finally:
        async_mongo_manager.client = None
        async_mongo_manager.database = None
