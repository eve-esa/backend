"""An agentic turn whose MCP tools/call is refused with 429 keeps going.

Agentic turns reach tools through the backend's own MCP proxy, which answers a
rate limited ``tools/call`` with a plain HTTP 429, not a JSON-RPC error. This
runs the real MCP client (``MultiServerMCPClient``, streamable HTTP) against an
in-process MCP server behind a 429 on ``tools/call``, then the real tools node
of the agents package: the model gets a ``Tool error`` message and the turn
does not raise.
"""

import asyncio
import json

import httpx
import pytest
from fastmcp import FastMCP
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

pytestmark = pytest.mark.no_db


def _mcp_app_refusing_tool_calls():
    server = FastMCP("echo")

    @server.tool
    def echo(text: str) -> str:
        return text

    app = server.http_app(path="/mcp", stateless_http=True, json_response=True)

    async def refusing(scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            await app(scope, receive, send)
            return
        body, more = b"", True
        while more:
            event = await receive()
            body += event.get("body", b"")
            more = event.get("more_body", False)
        try:
            method = json.loads(body).get("method")
        except ValueError:
            method = None
        if method == "tools/call":
            payload = json.dumps(
                {"detail": {"code": "rate_limited", "message": "Too many requests, retry in 5 s"}}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [[b"content-type", b"application/json"], [b"retry-after", b"5"]],
                }
            )
            await send({"type": "http.response.body", "body": payload})
            return
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await app(scope, replay, send)

    return app, refusing


async def test_rate_limited_tools_call_becomes_a_tool_error_not_a_dead_turn():
    from agents.graphs.base import AgentGraph

    app, refusing = _mcp_app_refusing_tool_calls()

    def client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=refusing),
            base_url="http://mcp.test",
            headers=headers,
            timeout=timeout or httpx.Timeout(10),
            auth=auth,
        )

    async with app.lifespan(app):
        client = MultiServerMCPClient(
            {
                "echo": {
                    "transport": "streamable_http",
                    "url": "http://mcp.test/mcp",
                    "httpx_client_factory": client_factory,
                }
            },
            tool_name_prefix=True,
        )
        tools = await client.get_tools(server_name="echo")
        assert [t.name for t in tools] == ["echo_echo"]

        tools_node = AgentGraph.make_tools_node(None, tools)
        call = AIMessage(
            content="",
            tool_calls=[{"name": "echo_echo", "args": {"text": "hi"}, "id": "call-1"}],
        )
        result = await asyncio.wait_for(tools_node({"messages": [call]}), timeout=20)

    [message] = result["messages"]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-1"
    assert message.content.startswith("Tool error")
