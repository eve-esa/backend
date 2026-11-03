# Backend agent guide

Earth Virtual Expert (EVE). This repository is the FastAPI backend. `main` is the only long-lived branch. Ship and access rules live in the [EVE Developer Guide](https://docs.google.com/document/d/1yZiQUSBL63g5I-A-0fS8DNC5wUl6U1YENwZKa25i0yY/edit) and the [EVE Platform Handbook](https://docs.google.com/document/d/1lVDCBiQ0DXZVba19XguXulZfXxPOBHigfGMCNoCHIcY/edit). The handbook’s legacy hostnames (`api.eve-chat.chat`, `staging-api.eve-chat.chat`, `agentic-api.eve-chat.chat`) are a different system. The live app is `dev.eve-chat.chat`, `staging.eve-chat.chat`, and `eve-chat.chat`, with the API under `/api`.

Before changing behaviour that a person can click, start the local stack in the order below and exercise it there. Deployed environments are for reading what already shipped.

## Architecture

One product, several processes:

- The browser loads the React app. Deployed, CloudFront serves that bundle from S3 and sends `/api/*` to this service. Locally the Vite dev server calls the API directly.
- This service owns conversations, generation, document retrieval (`POST /retrieve`), auth validation, and the MCP proxy (`/mcp/{server}`).
- The chat agent is a LangGraph graph from `eve-esa/agents` (import name `agents`). This repo runs the graph. It does not define it. `src/services/agents/graphs_bundle.py` prefers a checkout at `repos/eve-esa-agents` over the commit pinned in `requirements.txt`.
- Document search in an agent turn is the `eve_retrieval` MCP server. Its `retrieve` tool calls back into `POST /retrieve`. The URL the proxy dials is `config.url` on the `mcp_servers` row named `eve_retrieval`.
- Local dependencies in `docker-compose.yml`: MongoDB, Redis, Keycloak (realm `eve`, client `eve-frontend`), MinIO. Qdrant and the LLM endpoints stay on the URLs already in `.env`.

## Repositories

| Repository | What it is | Where a local agent uses it |
| --- | --- | --- |
| `eve-esa/backend` | This service | The workspace |
| `eve-esa/frontend` | React + Vite | Sibling checkout `../frontend` (the compose build context) |
| `eve-esa/agents` | LangGraph graphs (`react`, `simple`) | `repos/eve-esa-agents` inside this repo (`repos/` is gitignored) |
| `eve-esa/mcp-tool-registry` | MCP servers, including `servers/eve_retrieval` | A separate clone, run on the host |
| `eve-esa/backoffice` | Private ops UI (metrics and feedback) | Not part of the local stack |
| `eve-esa/infra` | Terraform for the four AWS accounts | Not part of the local stack |

Versions are independent. A frontend `v0.0.x` need not match a backend `v0.0.x`.

For a local test, frontend and agents are whatever `origin/main` is at the moment you start, not the commit pinned in `requirements.txt` and not a stale working tree.

```bash
git -C ../frontend fetch origin
git -C ../frontend switch main
git -C ../frontend pull --ff-only origin main

if [ ! -d repos/eve-esa-agents/.git ]; then
  git clone https://github.com/eve-esa/agents.git repos/eve-esa-agents
fi
git -C repos/eve-esa-agents fetch origin
git -C repos/eve-esa-agents switch main
git -C repos/eve-esa-agents pull --ff-only origin main
```

Stop if either pull is not fast-forward. The checkout has local commits, and discarding them is a human decision. `repos/eve-esa-agents/agents` must exist: `graphs_bundle.py` inserts that repo root at the front of `sys.path`.

## Local startup

Docker on this machine is Colima, and the binary is `docker-compose` (`docker compose` is not installed). If `docker info` cannot reach the socket, run `colima start`. If Colima says the profile is running but `~/.colima/default/docker.sock` is missing, run `colima stop --force && colima start`.

`.env` uses Docker hostnames (`MONGO_HOST=mongo`, `REDIS_URL=redis://redis:6379/0`). Leave that file as it is. Do not run `./start.sh` on the host against it, and do not print `.env`.

Run the steps in order. Skip a step when `docker-compose ps` or a listening port already shows that process.

### 1. Backend and its dependencies

```bash
docker-compose up -d backend
```

That also starts mongo, redis, keycloak, and minio. The source tree is bind-mounted and uvicorn reloads. After cloning or updating `repos/eve-esa-agents`, restart the backend container so the graph import runs again (`graphs_prefix` is cached for the life of the process).

- API: http://localhost:8001 (`8001:8000` in `docker-compose.yml`)
- API docs: http://localhost:8001/docs
- Keycloak: http://localhost:8080

Wait until `docker-compose ps` shows the backend running, then `curl -sf -o /dev/null -w '%{http_code}' http://localhost:8001/docs`.

If that stays empty and `docker exec backend-backend-1 python -c 'import server'` dies with SIGILL inside pymongo, leave the other containers up and run the API on the host. The frontend then uses `yarn dev` with no `VITE_API_URL` override, and the retrieval base URL below is `http://localhost:8000`:

```bash
LOAD_DOTENV_OVERRIDE=false MONGO_HOST=localhost REDIS_URL=redis://127.0.0.1:6379/0 \
  python3 -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Local `eve_retrieval`

The `mcp_servers` row in Mongo points at the deployed AgentCore runtime. That URL is not usable from this stack: the proxy would send the call to AgentCore, and local testing needs the retrieval server that calls this backend’s `POST /retrieve`.

Start `servers/eve_retrieval` from a clone of `eve-esa/mcp-tool-registry` on `origin/main` (same fetch / switch / `pull --ff-only` as the other repos). Port 8000 is the backend listener inside the container, so bind another port on the host. The server’s HTTP transport is streamable HTTP on `/mcp`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r servers/eve_retrieval/requirements.txt
EVE_API_BASE_URL=http://localhost:8001 \
  python servers/eve_retrieval/server.py --transport http --port 8090
```

`EVE_API_BASE_URL` is the API root with no `/api` suffix. Deployed environments mount the API under `/api`; this process does not. Leave the server running.

Then point the local Mongo row at that process. The backend container reaches the host as `host.docker.internal`. Read the database name from the backend container (`MONGO_DATABASE`) and substitute it for `<db>`:

```bash
docker-compose exec -T mongo mongosh -u root -p root --authenticationDatabase admin <db> --quiet --eval '
db.mcp_servers.updateOne(
  { name: "eve_retrieval", deleted_at: null },
  { $set: {
      "config.url": "http://host.docker.internal:8090/mcp",
      "config.transport": "streamable_http",
      enabled: true,
      updated_at: new Date()
  }}
)'
```

Confirm `matchedCount` is 1. Then `docker-compose restart backend`. The proxy caches `config.url` for `MCP_TOOLS_CACHE_TTL` (300 seconds), so a restart is what makes the new URL take effect. This edit is only the compose MongoDB volume.

### 3. Frontend

The compose `frontend` service sets `VITE_API_URL=http://localhost:8000`, which is not the published backend port. Start the sibling checkout on the host:

```bash
cd ../frontend
yarn install
VITE_API_URL=http://localhost:8001 yarn dev
```

Skip the `VITE_API_URL` override only when the API is the host process on port 8000. A variable already in the environment wins over `.env`. Leave this process running.

Open http://localhost:5173. The local Keycloak client allows `http://localhost:5173/*` as `redirect_uri`. Sign in as `dev@eve.example.com` / `eve-dev-password`.

### 4. Check the path you changed

Docs respond, the frontend loads, sign-in returns to the app, and a chat turn that should retrieve documents calls the local `eve_retrieval` (its log shows `POST /retrieve` against `localhost:8001`). Unit tests stay `pytest -m "not e2e"` with `FREE_TOKENS=100` and `FREE_PERIOD_MONTHS=1`.

## Observability

Three places, from the developer guide and the handbook. They show deployed traffic. A local run does not appear there unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set; empty means telemetry stays off.

| Platform | Use it for | URL |
| --- | --- | --- |
| Langfuse | Agent traces | https://langfuse.app.eve-chat.chat |
| HyperDX | Session replay, errors, backend logs | https://o11y.app.eve-chat.chat |
| Backoffice | The same signals, plus the metrics and feedback that matter day to day | https://backoffice.dev.eve-chat.chat, `backoffice.staging.eve-chat.chat`, `backoffice.eve-chat.chat` |

Traces reach Langfuse through the OpenTelemetry collector (`src/observability`). Thumbs become Langfuse scores only when `FEATURE_LANGFUSE_SCORES` and the three `LANGFUSE_*` variables are set. The frontend ships HyperDX browser instrumentation (`@hyperdx/browser`). `service.name` for this process is `eve-backend`. Dev, staging, and prod all export to the same ClickHouse and are told apart by `deployment.environment.name`.

Ask a deployed environment what it is running before comparing a trace to the code you have open:

```bash
curl -s https://dev.eve-chat.chat/api/health
```

### Query HyperDX from this workspace

When someone asks to diagnose deployed logs, errors, traces, or sessions, query the production ClickHouse that backs HyperDX. The MCP server `mcp-clickhouse` is configured in `.cursor/mcp.json` (gitignored). Its tools are `list_databases`, `list_tables`, and `run_query`. Queries are read-only. Use `run_query`. Do not print `CLICKHOUSE_PASSWORD`, and do not copy it into this file, into `.env`, or into a command line.

`CLICKHOUSE_HOST=127.0.0.1` and port `8123` are the production server, reached through a Session Manager tunnel. `o11y.app.eve-chat.chat` is the HyperDX UI, not the ClickHouse port. If `run_query` fails with connection refused, open the tunnel and leave it running:

```bash
HOST=$(aws ssm get-parameter --profile esa-eve-prod --region eu-west-1 \
  --name /esa-eve/prod/o11y/instance_id --query Parameter.Value --output text)
aws ssm start-session --profile esa-eve-prod --region eu-west-1 \
  --target "$HOST" --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8123"],"localPortNumber":["8123"]}'
```

The database is `default`. The tables HyperDX uses are `otel_logs`, `otel_traces`, and `hyperdx_sessions`. Filter on `ServiceName = 'eve-backend'` and `ResourceAttributes['deployment.environment.name']` (or the same attribute on logs) before reading rows. Summarize counts and the failing lines. Do not dump a full table into the chat. The login is the ClickHouse user `default`; the password is the `default_password` key of Secrets Manager secret `esa-eve/prod/o11y/clickhouse`, already stored in `.cursor/mcp.json`.

## Leave alone

- `.env`, published ports, and Keycloak realm files. A stale `AUTH_*` in `.env` wins over `docker-compose.yml` because dotenv loads with `override=True`.
- The compose `frontend` service, unless someone explicitly asks for that container.
- `CHANGELOG.md` and `v*` tags. Release-please owns both.
- The AgentCore URL that was in `mcp_servers` before the local rewrite. Put it back only when the local volume is thrown away.
- `.cursor/mcp.json`. It holds the ClickHouse password and stays out of git.
