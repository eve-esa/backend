"""Black-box integration test for the agentic message-generation endpoint.

This test authenticates as a real user over HTTP and exercises
``POST /conversations/{id}/generate-agentic`` end-to-end against a running
backend — exactly like ``testing/load_test_agentic.py`` does:

    1. POST {BASE_URL}/login           with {email, password}  → access_token
    2. POST {BASE_URL}/conversations   with the bearer token   → conv_id
    3. POST {BASE_URL}/conversations/{conv_id}/generate-agentic
         with the new request shape (public_mcp_servers, agent, ...)

It hits the live LLM and the configured ``geocode`` MCP server.

Required env:
    BASE_URL   e.g. ``http://127.0.0.1:8000`` (inside the backend container)
               or  ``https://staging-agentic-api.eve-chat.chat``
    EMAIL      existing user email
    PASSWORD   that user's password

Optional env:
    AGENTIC_TEST_QUERY   override the prompt (default: geocode Huntsville)
    AGENTIC_REAL_TIMEOUT per-request HTTP timeout in seconds (default: 180)
"""

import logging
import os
import time

import httpx
import pytest

logger = logging.getLogger(__name__)


def _extract_tool_calls(trace) -> list[str]:
    """Walk an agentic trace and collect every tool name that was invoked.

    Mirrors the heuristic used in ``testing/load_test_agentic.py`` — the trace
    structure isn't formally documented, so this walks all nested dicts/lists
    and collects names from anything that looks like a tool-call record.
    """

    names: list[str] = []

    def _walk(node) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            step_type = str(node.get("type", "")).lower()
            if "tool" in step_type:
                for key in ("tool_name", "name", "tool", "function_name"):
                    val = node.get(key)
                    if isinstance(val, str):
                        names.append(val)
                        break

            if "tool_calls" in node:
                for tc in node.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        for key in ("name", "tool_name", "tool", "function_name"):
                            val = tc.get(key) or (tc.get("function") or {}).get("name")
                            if isinstance(val, str):
                                names.append(val)
                                break

            for v in node.values():
                if isinstance(v, (dict, list)):
                    _walk(v)

    _walk(trace)
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


@pytest.mark.asyncio
async def test_generate_agentic_huntsville_real_end_to_end():
    """Black-box end-to-end test that authenticates as a real user and hits
    ``/generate-agentic`` over HTTP — no ASGI in-process, no mocking.

    The agent decides on its own whether to call the geocode tool, the answer
    comes from the live LLM, and assertions are intentionally loose so that
    minor wording changes from the model don't cause flakes.
    """

    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    email = os.getenv("EMAIL", "").strip()
    password = os.getenv("PASSWORD", "").strip()

    missing = [
        name
        for name, value in (("EMAIL", email), ("PASSWORD", password))
        if not value
    ]
    if missing:
        pytest.skip(
            "Live agentic test requires the following env vars to be set: "
            f"{', '.join(missing)} (BASE_URL also; defaults to "
            "http://127.0.0.1:8000)."
        )

    timeout_s = float(os.getenv("AGENTIC_REAL_TIMEOUT", "180"))
    query = os.getenv(
        "AGENTIC_TEST_QUERY",
        "Please use the geocode tool to find the latitude and longitude "
        "of Huntsville, Alabama (United States).",
    )

    logger.info(
        "=== Live agentic test against %s ===\n\n"
        "user:    %s\n"
        "timeout: %.0fs",
        base_url,
        email,
        timeout_s,
    )

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        # 1) Login — mirrors load_test_agentic.login()
        login_resp = await client.post(
            f"{base_url}/login",
            json={"email": email, "password": password},
        )
        assert login_resp.status_code == 200, (
            f"Login failed ({login_resp.status_code}): {login_resp.text}"
        )
        token = login_resp.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {token}"}
        logger.info("Login OK — token (truncated): %s…", token[:40])

        conv_id = None
        try:
            # 2) Create conversation — mirrors load_test_agentic.create_conversation()
            conv_resp = await client.post(
                f"{base_url}/conversations",
                json={"name": "Agentic Geocode E2E Test"},
                headers=auth_headers,
            )
            assert conv_resp.status_code == 200, (
                f"create_conversation failed: {conv_resp.text}"
            )
            conv_id = conv_resp.json()["id"]
            logger.info("Created conversation: %s", conv_id)

            # 3) Call generate-agentic with the new request shape.
            payload = {
                "query": query,
                "temperature": 0.1,
                "max_new_tokens": 8192,
                "public_mcp_servers": ["geocode"],
                "agent": "react",
                "public_collections": [],
            }
            logger.info(
                "=== POST %s/conversations/%s/generate-agentic ===\n\n%s",
                base_url,
                conv_id,
                payload,
            )

            t0 = time.perf_counter()
            resp = await client.post(
                f"{base_url}/conversations/{conv_id}/generate-agentic",
                json=payload,
                headers=auth_headers,
            )
            wall_s = time.perf_counter() - t0

            logger.info("HTTP status: %s  wall time: %.2fs", resp.status_code, wall_s)
            assert resp.status_code == 200, (
                f"Non-200 response ({wall_s:.1f}s): {resp.text}"
            )

            body = resp.json()
            answer = body.get("answer") or ""
            documents = body.get("documents") or []
            trace = body.get("trace")
            latencies = (body.get("metadata") or {}).get("latencies") or {}
            tool_calls = _extract_tool_calls(trace)

            logger.info("=== answer (%d chars) ===\n\n%s", len(answer), answer)
            logger.info(
                "=== tool calls extracted from trace ===\n\n%s", tool_calls
            )
            logger.info("=== server-reported latencies ===\n\n%s", latencies)
            logger.info("documents: %d entries", len(documents))


            # Loose assertions — minor wording changes shouldn't fail the test.
            assert isinstance(answer, str) and answer.strip(), (
                "Agent returned empty answer"
            )
            assert body.get("conversation_id") == conv_id

            lower = answer.lower()
            assert "huntsville" in lower, (
                "Answer doesn't mention 'Huntsville'. Likely the agent didn't "
                f"run the geocode tool or the LLM lost the city name.\n"
                f"Answer:\n{answer}"
            )

            # When the trace exposes tool calls, at least one should be geocode.
            # Graph implementations vary in how richly they record steps, so we
            # don't fail when the trace is opaque.
            if tool_calls:
                assert any("geocode" in tc.lower() for tc in tool_calls), (
                    f"Trace shows tool calls {tool_calls} but none are 'geocode'"
                )
        finally:
            if conv_id:
                try:
                    await client.delete(
                        f"{base_url}/conversations/{conv_id}",
                        headers=auth_headers,
                    )
                except Exception:
                    logger.exception("Conversation cleanup failed (non-fatal)")
