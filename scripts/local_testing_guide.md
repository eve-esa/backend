# Local Testing Guide

This guide covers the local Docker backend, MongoDB, the agentic test script,
and MCP server setup used for local `generate-agentic` testing.

## Backend URLs

Start or restart the backend stack from the repo root:

```bash
docker compose -f backend/docker-compose.yml up -d backend frontend
```

## MongoDB Compass

Mongo is exposed on the host. Use this connection string:

```text
mongodb://root:root@localhost:27017/eve-backend?authSource=admin
```

## Login Token

You can let `test_generate_agentic.py` log in for you with `--email` and
`--password`. If you need a token manually:

```bash
curl -sS -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@gmail.com","password":"testtesttest"}'
```

The response contains `access_token` and `refresh_token`.

## Agentic Test Script


```bash
python3 scripts/test_generate_agentic.py \
  --email test@gmail.com \
  --password testtesttest \
  --conversation-name "My agentic test" \
  --llm-type main \
  "Hello, reply with one short sentence."
```

The script:

- logs in unless `--token` is provided
- creates a conversation unless `--conversation-id` is provided
- sends one message to `/conversations/{id}/generate-agentic`
- prints `message_id`, `conversation_id`, `use_rag`, tool calls, latencies, and answer
- keeps newly created conversations by default

Use an existing conversation:

```bash
python3 scripts/test_generate_agentic.py \
  --email test@gmail.com \
  --password testtesttest \
  --conversation-id 6a7581978836e6f03faa8693 \
  --llm-type main \
  "Continue this conversation."
```

Delete a created conversation after the test:

```bash
python3 scripts/test_generate_agentic.py \
  --email test@gmail.com \
  --password testtesttest \
  --delete-conversation \
  "Temporary test."
```

## Classic RAG vs Agentic RAG

Classic `/messages` uses the `should_use_rag()` classifier and vector retrieval.
Classic vector retrieval runs only when all of these are true:

```python
len(request.collection_ids) > 0
and request.k > 0
and rag_decision_result.use_rag
```

Agentic `/generate-agentic` is different. It does not use `should_use_rag()`.
For agentic, `use_rag=True` means the LangGraph agent produced at least one
`ToolMessage`. In practice, this requires an MCP server to be selected and the
agent to call one of its tools.

Public collection flags are accepted by the agentic endpoint, but they do not
create a vector-search tool by themselves. They are useful for classic RAG, and
they can be passed through to an MCP retrieval tool if that tool accepts them.

## MCP Server Registration

The backend does not auto-discover MCP server code or running MCP processes.
An MCP server must be registered in MongoDB before `--mcp-server <name>` works.


## Running `eve_retrieval` MCP Locally

Start the EVE retrieval MCP server from a separate terminal:

```bash
cd /Users/jinorohit/pischool/mcp_server/tools

python3 -m venv .venv
source .venv/bin/activate
pip install -r servers/eve_retrieval/requirements.txt

EVE_API_BASE_URL=http://localhost:8000 \
EVE_EMAIL=test@gmail.com \
EVE_PASSWORD=testtesttest \
python servers/eve_retrieval/server.py --transport http --port 9100
```

Leave this process running. Verify it is listening:

Register only `eve_retrieval` and disable other MCP servers:


Test an agentic retrieval query:

```bash
python3 scripts/test_generate_agentic.py \
  --email test@gmail.com \
  --password testtesttest \
  --conversation-name "Agentic eve_retrieval test" \
  --llm-type main \
  --mcp-server eve_retrieval \
  --max-new-tokens 2048 \
  "Use the eve_retrieval_retrieve tool to retrieve documents for: What is ESA?"
```