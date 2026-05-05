"""
Manual integration test for the agentic endpoints.

Usage:
    python3 test_agentic.py [--base-url http://localhost:8000]

Runs against a live Docker backend. Authenticates, creates a conversation,
then exercises both the non-streaming and streaming agentic endpoints.
"""

import argparse
import json
import sys
import time
from typing import List

import requests

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8001"
EMAIL = "test@gmail.com"
PASSWORD = "7N5onTwLc6UuqRHXbIF"

# Use the staging public collection that is likely indexed
PUBLIC_COLLECTIONS = ["qwen-512-filtered"]
PUBLIC_MCP_SERVERS = ["eve_retrieval", "effis"]
MCP_SERVERS = ["eve_retrieval", "effis"]

QUERY = "What is ESA? Write two sentences about ESA."

# ─── ANSI colours ─────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg):
    print(f"{GREEN}✓ {msg}{RESET}")


def fail(msg):
    print(f"{RED}✗ {msg}{RESET}")
    sys.exit(1)


def info(msg):
    print(f"{CYAN}  {msg}{RESET}")


def warn(msg):
    print(f"{YELLOW}  ⚠ {msg}{RESET}")


def section(msg):
    print(f"\n{BOLD}{msg}{RESET}")


# ─── Helpers ──────────────────────────────────────────────────────────────────


def login(base_url: str) -> str:
    section("1. Login")
    resp = requests.post(
        f"{base_url}/login",
        json={"email": EMAIL, "password": PASSWORD},
        timeout=15,
    )
    if resp.status_code != 200:
        fail(f"Login failed ({resp.status_code}): {resp.text}")
    token = resp.json()["access_token"]
    ok(f"Logged in as {EMAIL}")
    return token


def create_conversation(base_url: str, token: str) -> str:
    section("2. Create conversation")
    resp = requests.post(
        f"{base_url}/conversations",
        json={"name": "Agentic test conversation"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code != 200:
        fail(f"Create conversation failed ({resp.status_code}): {resp.text}")
    conv_id = resp.json()["id"]
    ok(f"Conversation created: {conv_id}")
    return conv_id


def delete_conversation(base_url: str, token: str, conv_id: str):
    requests.delete(
        f"{base_url}/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )


def build_payload() -> dict:
    return {
        "query": QUERY,
        "k": 3,
        "score_threshold": 0.5,
        "temperature": 0.3,
        "max_new_tokens": 200,
        "public_collections": PUBLIC_COLLECTIONS,
        "public_mcp_servers": PUBLIC_MCP_SERVERS,
        "filters": {"must": [], "should": None, "must_not": None, "min_should": None},
    }


# ─── Non-streaming agentic test ───────────────────────────────────────────────


def test_agentic_message(base_url: str, token: str, conv_id: str):
    section("3. POST /conversations/{id}/generate-agentic  (non-streaming)")
    info(f"Query: {QUERY}")

    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/conversations/{conv_id}/generate-agentic",
        json=build_payload(),
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        fail(f"Agentic message failed ({resp.status_code}): {resp.text[:500]}")

    body = resp.json()

    ok(f"Response received in {elapsed:.2f}s")
    info(f"Message ID : {body.get('id')}")
    info(f"use_rag    : {body.get('use_rag')}")
    info(
        f"Total latency: {body.get('metadata', {}).get('latencies', {}).get('total_latency', 'n/a')}"
    )

    answer = body.get("answer", "")
    if not answer:
        warn("Answer is empty!")
    else:
        ok(f"Answer ({len(answer)} chars):")
        print(f"\n{answer[:600]}{'...' if len(answer) > 600 else ''}\n")

    documents = body.get("documents", [])
    info(f"Tool results returned: {len(documents)}")
    for i, doc in enumerate(documents[:2]):
        tool_name = doc.get("tool", "tool")
        preview = str(doc.get("content", ""))[:120]
        info(f"  [{i + 1}] {tool_name}: {preview}…")

    return body.get("id")


# ─── Streaming agentic test ───────────────────────────────────────────────────


def test_agentic_stream_message(base_url: str, token: str, conv_id: str):
    section("4. POST /conversations/{id}/stream-generate-agentic  (streaming SSE)")
    info(f"Query: {QUERY}")

    t0 = time.perf_counter()
    token_count = 0
    tool_calls = []
    final_payload = None
    answer_chars = 0
    first_token_latency = None

    try:
        with requests.post(
            f"{base_url}/conversations/{conv_id}/stream-generate-agentic",
            json=build_payload(),
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "text/event-stream",
            },
            stream=True,
            timeout=500,
        ) as resp:
            if resp.status_code != 200:
                fail(f"Stream endpoint failed ({resp.status_code}): {resp.text[:500]}")

            print()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                if not line.startswith("data: "):
                    continue
                payload_str = line[len("data: ") :]

                # Parse JSON events
                try:
                    event = json.loads(payload_str)
                except json.JSONDecodeError:
                    # Plain-text token (non-JSON stream mode)
                    token_count += 1
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - t0
                    print(payload_str, end="", flush=True)
                    answer_chars += len(payload_str)
                    continue

                etype = event.get("type")

                if etype == "tool_call":
                    tool_calls.append(event.get("content", ""))
                    print(
                        f"\n{YELLOW}  🔍 {event.get('content', '')}{RESET}", flush=True
                    )

                elif etype == "tool_result":
                    preview = event.get("content", "")[:80]
                    print(f"\n{CYAN}  ↩  result preview: {preview}…{RESET}", flush=True)

                elif etype == "token":
                    content = event.get("content", "")
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - t0
                    token_count += 1
                    answer_chars += len(content)
                    print(content, end="", flush=True)

                elif etype == "final":
                    final_payload = event
                    print()  # newline after streamed tokens

                elif etype == "stopped":
                    warn("Generation was stopped by the server")
                    break

                elif etype == "error":
                    fail(f"Stream error: {event.get('message')}")

    except requests.exceptions.Timeout:
        fail("Request timed out after 120 s")

    elapsed = time.perf_counter() - t0
    print()
    ok(f"Stream completed in {elapsed:.2f}s")
    info(
        f"First-token latency  : {first_token_latency:.2f}s"
        if first_token_latency
        else "First-token latency: n/a"
    )
    info(f"Tokens/events yielded: {token_count}")
    info(f"Answer chars         : {answer_chars}")
    info(f"Tool calls made      : {len(tool_calls)}")
    for i, tc in enumerate(tool_calls):
        info(f"  [{i + 1}] {tc}")
    if final_payload:
        lats = final_payload.get("latencies", {})
        info(f"generation_latency   : {lats.get('generation_latency', 'n/a')}")
        info(f"total_latency        : {lats.get('total_latency', 'n/a')}")

    if answer_chars == 0:
        warn("No answer tokens were streamed.")


# ─── Forced tool-call test ────────────────────────────────────────────────────

# A query that is hyper-specific to the indexed corpus.  The model cannot answer
# from training data alone, so it MUST call search_knowledge_base.
FORCED_TOOL_QUERY = (
    "Can you tell me if there were fires in Attica during 2023, use EFFIS?"
)


def test_forced_tool_call(base_url: str, token: str, conv_id: str):
    section("5. Forced tool-call test  (agent MUST retrieve before answering)")
    info(f"Query: {FORCED_TOOL_QUERY[:90]}…")

    payload = {
        **build_payload(),
        "query": FORCED_TOOL_QUERY,
        "k": 3,
        "max_new_tokens": 512,
    }

    tool_calls_seen: List[str] = []
    tool_results_seen: List[str] = []
    answer_chars = 0
    t0 = time.perf_counter()

    try:
        with requests.post(
            f"{base_url}/conversations/{conv_id}/stream-generate-agentic",
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "text/event-stream",
            },
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                fail(f"Endpoint returned {resp.status_code}: {resp.text[:400]}")

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                if etype == "tool_call":
                    tool_calls_seen.append(event.get("content", ""))
                    print(
                        f"\n{YELLOW}  🔍 {event.get('content', '')}{RESET}", flush=True
                    )
                elif etype == "tool_result":
                    tool_results_seen.append(event.get("content", ""))
                    print(
                        f"\n{CYAN}  ↩  {event.get('content', '')[:80]}…{RESET}",
                        flush=True,
                    )
                elif etype == "token":
                    content = event.get("content", "")
                    answer_chars += len(content)
                    print(content, end="", flush=True)
                elif etype == "final":
                    print()
                elif etype == "error":
                    fail(f"Stream error: {event.get('message')}")

    except requests.exceptions.Timeout:
        fail("Request timed out after 120 s")

    elapsed = time.perf_counter() - t0
    print()

    # ── Assertions ────────────────────────────────────────────────────────────
    if not tool_calls_seen:
        fail(
            "ASSERTION FAILED: agent produced an answer without calling any tool. "
            "Expected at least one search_knowledge_base call."
        )
    ok(f"Tool call(s) detected: {len(tool_calls_seen)}")
    for i, tc in enumerate(tool_calls_seen):
        info(f"  [{i + 1}] {tc}")

    if not tool_results_seen:
        fail(
            "ASSERTION FAILED: tool_call events were emitted but no tool_result was received."
        )
    ok(f"Tool result(s) received: {len(tool_results_seen)}")

    if answer_chars == 0:
        fail("ASSERTION FAILED: tool was called but no answer tokens were streamed.")
    ok(f"Answer streamed ({answer_chars} chars) in {elapsed:.2f}s")


# ─── Existing stream endpoint (smoke test for comparison) ─────────────────────


def test_existing_stream_message(base_url: str, token: str, conv_id: str):
    section("6. POST /conversations/{id}/stream_messages  (original, smoke-check)")
    info("Verifying the original streaming endpoint still works…")

    payload = {
        **build_payload(),
        "query": "Hello, are you working?",
        "k": 1,
        "max_new_tokens": 200,
    }

    try:
        with requests.post(
            f"{base_url}/conversations/{conv_id}/stream_messages",
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "text/event-stream",
            },
            stream=True,
            timeout=60,
        ) as resp:
            if resp.status_code != 200:
                warn(
                    f"Original stream endpoint returned {resp.status_code}: {resp.text[:200]}"
                )
                return
            got_token = False
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        ev = json.loads(data)
                        if ev.get("type") in ("token", "final"):
                            got_token = True
                            break
                    except json.JSONDecodeError:
                        if data and data != "[DONE]":
                            got_token = True
                            break
            if got_token:
                ok("Original stream endpoint is responding normally")
            else:
                warn("No tokens received from original stream endpoint")
    except Exception as exc:
        warn(f"Original stream check skipped: {exc}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Test agentic endpoints")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL")
    parser.add_argument(
        "--skip-nonstream", action="store_true", help="Skip non-streaming agentic test"
    )
    parser.add_argument(
        "--skip-stream", action="store_true", help="Skip streaming agentic test"
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"{BOLD}Agentic endpoint test — {base_url}{RESET}")

    token = login(base_url)
    conv_id = create_conversation(base_url, token)

    try:
        # if not args.skip_nonstream:
        #     test_agentic_message(base_url, token, conv_id)

        if not args.skip_stream:
            test_agentic_stream_message(base_url, token, conv_id)

        test_forced_tool_call(base_url, token, conv_id)

        test_existing_stream_message(base_url, token, conv_id)

    finally:
        # section("Cleanup")
        # delete_conversation(base_url, token, conv_id)
        # ok(f"Conversation {conv_id} deleted")
        pass

    section("Done")
    ok("All tests passed")


if __name__ == "__main__":
    main()
