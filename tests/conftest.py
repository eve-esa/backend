# tests/conftest.py
import os
import sys
import warnings

# Silence import-time noise from third-party deps before they're loaded
# (pytest's ini filterwarnings only applies post-collection, so these wouldn't
# otherwise be suppressed for imports triggered by ``from server import app``).
warnings.filterwarnings(
    "ignore",
    message=r".*datetime\.datetime\.utcnow\(\).*",
    category=DeprecationWarning,
)

# langchain_core installs its own warning filters at import time that take
# precedence over generic ``Warning`` filters, so we import the exact
# category and silence it explicitly. Done before any langgraph import so the
# ``allowed_objects`` PendingDeprecationWarning never surfaces.
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    warnings.simplefilter("ignore", LangChainPendingDeprecationWarning)
except Exception:  # pragma: no cover - defensive: langchain may move the class
    pass

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from server import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.mongo import async_mongo_manager
from src.database.indexes import ensure_indexes
from src.services.provider_catalog import (
    clear_provider_catalog_cache_for_tests,
    ensure_provider_catalog_seeded,
)


def _resolve_test_mongo_uri() -> str:
    """Pick the best Mongo URI to use for tests.

    Resolution order:
    1. ``TEST_MONGO_URI`` — explicit override for tests.
    2. ``MONGO_URI`` — generic override (legacy).
    3. Per-host env (``MONGO_HOST`` / ``MONGO_USERNAME`` / ``MONGO_PASSWORD`` /
       ``MONGO_PORT``) targeting a dedicated ``eve_backend_test`` database — this
       keeps tests isolated from the prod ``MONGO_DATABASE`` collection.
    4. ``mongodb://localhost:27017/eve_backend_test`` as a last-resort default.
    """

    for env_key in ("TEST_MONGO_URI", "MONGO_URI"):
        value = os.getenv(env_key)
        if value:
            return value

    host = os.getenv("MONGO_HOST", "").strip()
    if host:
        user = os.getenv("MONGO_USERNAME", "").strip()
        password = os.getenv("MONGO_PASSWORD", "").strip()
        port = os.getenv("MONGO_PORT", "27017").strip() or "27017"
        auth = f"{user}:{password}@" if user else ""
        auth_source_qs = "?authSource=admin" if user else ""
        return f"mongodb://{auth}{host}:{port}/eve_backend_test{auth_source_qs}"

    return "mongodb://localhost:27017/eve_backend_test"


@pytest_asyncio.fixture(autouse=True)
async def _db_connection():
    """Connect to MongoDB before each test and close afterwards.

    The connection string is resolved via :func:`_resolve_test_mongo_uri`, which
    prefers explicit overrides and otherwise reuses the same host/credentials
    as the running app but always targets the dedicated ``eve_backend_test``
    database so test runs don't pollute production data.
    """

    connection_string = _resolve_test_mongo_uri()
    await async_mongo_manager.connect(connection_string)
    await ensure_indexes()
    await ensure_provider_catalog_seeded()
    clear_provider_catalog_cache_for_tests()

    try:
        yield
    finally:
        await async_mongo_manager.close()
        async_mongo_manager.database = None


@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
