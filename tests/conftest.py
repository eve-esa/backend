# tests/conftest.py
import os
import sys
import asyncio

import pytest
import pytest_asyncio
from httpx import AsyncClient
from server import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.mongo import async_mongo_manager


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def _db_connection():
    """Connect to MongoDB before each test and close afterwards.

    Any MongoModel instances created during the test that are registered via
    ``tests.model_tracker.register_model`` will be deleted automatically at the
    end of the test.
    """

    connection_string = (
        os.getenv("MONGO_URI") or "mongodb://localhost:27017/eve_backend_test"
    )

    await async_mongo_manager.connect(connection_string)

    try:
        yield
    finally:
        await async_mongo_manager.close()


@pytest_asyncio.fixture(scope="session")
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
