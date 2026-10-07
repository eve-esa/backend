#!/usr/bin/env python3
"""
Examples:
    python3 backend/scripts/test_generate_agentic.py \
      --email test@gmail.com --password testtesttest \
      "What are Sentinel-2 applications?"

    python3 backend/scripts/test_generate_agentic.py \
      --token "$ACCESS_TOKEN" --mcp-server geocode \
      "Use the geocode tool to find Huntsville, Alabama."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from urllib import error, request


def _json_request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: float,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed: dict[str, Any] | list[Any] | str = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed


def _extract_tool_calls(trace: Any) -> list[str]:
    calls: list[str] = []
    if not isinstance(trace, list):
        return calls
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        for tool_call in entry.get("tool_calls") or []:
            if isinstance(tool_call, dict) and tool_call.get("name"):
                calls.append(str(tool_call["name"]))
        if entry.get("role") == "tool" and entry.get("name"):
            calls.append(str(entry["name"]))
    return calls


def _parse_csv(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a conversation and call /generate-agentic."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--email", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--token", default=None, help="Existing access token.")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Message query to send to /generate-agentic.",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Use an existing conversation. Omit to create a new one.",
    )
    parser.add_argument("--conversation-name", default="Agentic smoke test")
    parser.add_argument(
        "--query",
        dest="query_flag",
        default="Hello, this is an agentic backend smoke test. Reply with one short sentence.",
        help="Message query. Kept for backward compatibility; positional query is preferred.",
    )
    parser.add_argument("--llm-type", default="main")
    parser.add_argument("--agent", default="react")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--public-collection",
        action="append",
        default=[],
        help="Public collection name. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--mcp-server",
        action="append",
        default=[],
        help="MCP server name. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--delete-conversation",
        action="store_true",
        help="Delete the conversation created by this script after the test.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    token = args.token
    query = args.query or args.query_flag

    if not token:
        if not args.email or not args.password:
            print("Provide either --token or both --email and --password.", file=sys.stderr)
            return 2
        status, body = _json_request(
            "POST",
            f"{base_url}/login",
            payload={"email": args.email, "password": args.password},
            timeout=args.timeout,
        )
        if status != 200 or not isinstance(body, dict) or not body.get("access_token"):
            print(f"Login failed ({status}): {json.dumps(body, indent=2)}", file=sys.stderr)
            return 1
        token = str(body["access_token"])
        print(f"Login OK: token starts with {token[:12]}...")

    created_conversation = False
    conversation_id = args.conversation_id
    if not conversation_id:
        status, body = _json_request(
            "POST",
            f"{base_url}/conversations",
            payload={"name": args.conversation_name},
            token=token,
            timeout=args.timeout,
        )
        if status != 200 or not isinstance(body, dict) or not body.get("id"):
            print(
                f"Create conversation failed ({status}): {json.dumps(body, indent=2)}",
                file=sys.stderr,
            )
            return 1
        conversation_id = str(body["id"])
        created_conversation = True
        print(f"Created conversation: {conversation_id}")

    payload = {
        "query": query,
        "llm_type": args.llm_type,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "k": args.k,
        "public_collections": _parse_csv(args.public_collection),
        "public_mcp_servers": _parse_csv(args.mcp_server),
        "agent": args.agent,
    }

    print(f"POST {base_url}/conversations/{conversation_id}/generate-agentic")
    print(json.dumps(payload, indent=2))

    started = time.perf_counter()
    status, body = _json_request(
        "POST",
        f"{base_url}/conversations/{conversation_id}/generate-agentic",
        payload=payload,
        token=token,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - started

    print(f"\nHTTP {status} in {elapsed:.2f}s")
    if status != 200 or not isinstance(body, dict):
        print(json.dumps(body, indent=2) if not isinstance(body, str) else body)
        return 1

    answer = str(body.get("answer") or "")
    trace = body.get("trace")
    documents = body.get("documents") or []
    latencies = (body.get("metadata") or {}).get("latencies") or {}
    tool_calls = _extract_tool_calls(trace)

    print(f"message_id: {body.get('id')}")
    print(f"conversation_id: {body.get('conversation_id')}")
    print(f"use_rag: {body.get('use_rag')}")
    print(f"documents/tool_results: {len(documents)}")
    print(f"tool_calls: {tool_calls}")
    print(f"latencies: {json.dumps(latencies, indent=2)}")
    print("\nanswer:\n" + answer)

    if created_conversation and args.delete_conversation:
        status, cleanup_body = _json_request(
            "DELETE",
            f"{base_url}/conversations/{conversation_id}",
            token=token,
            timeout=args.timeout,
        )
        if status in (200, 204):
            print(f"\nDeleted conversation: {conversation_id}")
        else:
            print(
                f"\nConversation cleanup failed ({status}): "
                f"{json.dumps(cleanup_body, indent=2)}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
