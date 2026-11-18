"""A user row and a global row with one name: the proxy resolves the caller's row.

Without a sort ``find_one`` returns whichever row the store reads first, so a
turn could route a global server's calls to the user's URL or the reverse.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.database.mongo import get_collection
from src.middlewares.auth import AUTH_TYPE_API_KEY, Principal
from src.routers import mcp_proxy
from src.routers.mcp_proxy import MCPProxyDispatcher

_PROXY = "src.routers.mcp_proxy"
_NAME = "row-precedence-test"
_USER_ID = "6ac1bbbbbbbbbbbbbbbbbbbb"


async def _resolved_url(user_id: str) -> str:
    built: list = []

    async def _build(url, _provider):
        built.append(url)
        return object()

    principal = Principal(user_id, AUTH_TYPE_API_KEY, "key-1")
    scope = {"headers": [(b"authorization", b"Bearer eve_test-token")]}
    with patch(
        f"{_PROXY}.resolve_principal_from_bearer_token", AsyncMock(return_value=principal)
    ), patch(f"{_PROXY}.get_cognito_token_provider", return_value=MagicMock()), patch(
        f"{_PROXY}.build_proxy_app", _build
    ):
        await MCPProxyDispatcher(main_app=None)._resolve_proxy(scope, _NAME)
    return built[0]


async def test_the_callers_row_wins_over_the_global_row_in_either_insert_order():
    collection = get_collection("mcp_servers")
    rows = {
        "global": {"name": _NAME, "enabled": True, "user_id": None,
                   "config": {"url": "https://global.invalid/mcp"}},
        "user": {"name": _NAME, "enabled": True, "user_id": _USER_ID,
                 "config": {"url": "https://user.invalid/mcp"}},
    }
    try:
        for order in (("global", "user"), ("user", "global")):
            await collection.delete_many({"name": _NAME})
            mcp_proxy._server_url_cache.clear()
            for key in order:
                await collection.insert_one(dict(rows[key]))

            assert await _resolved_url(_USER_ID) == "https://user.invalid/mcp"
            mcp_proxy._server_url_cache.clear()
            # Another user sees only the global row.
            assert await _resolved_url("6ac1cccccccccccccccccccc") == "https://global.invalid/mcp"
    finally:
        await collection.delete_many({"name": _NAME})
        mcp_proxy._server_url_cache.clear()
