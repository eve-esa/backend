import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "https://custom-agents-dev-mcp.scholargateway.ai"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def env(key: str) -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def post(url: str, payload: dict | None = None, headers: dict | None = None):
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"User-Agent": UA, **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, dict(resp.headers.items()), resp.read().decode()


def main():
    auth = env("WILEY_AUTH_TOKEN")
    if not auth:
        raise SystemExit(f"WILEY_AUTH_TOKEN missing from {ROOT / '.env'}")

    status, _, body = post(
        f"{HOST}/oauth2/token?grant_type=client_credentials",
        headers={"Authorization": auth, "Accept": "application/json"},
    )
    token = json.loads(body)["access_token"]
    print("oauth", status)

    mcp_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    status, headers, body = post(
        f"{HOST}/mcp",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "wiley-mcp-healthcheck", "version": "1.0"},
            },
        },
        headers=mcp_headers,
    )
    print("initialize", status)
    print(body)
    session = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")
    if session:
        mcp_headers["Mcp-Session-Id"] = session

    status, _, body = post(
        f"{HOST}/mcp",
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "semanticSearch",
                "arguments": {
                    "query": "What are the impacts of climate change on Arctic sea ice?",
                    "topN": 5,
                },
            },
        },
        headers=mcp_headers,
    )
    print("semanticSearch", status)
    print(body)


if __name__ == "__main__":
    main()
