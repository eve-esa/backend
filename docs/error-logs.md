# Error logging

Ops log of failures, stored in MongoDB collection `error_logs`. The
backoffice reads this collection. When Langfuse keys are set, agentic
`kind`s are also child events on the generation trace; without keys that
path is a no-op (hosted ECS today has no Langfuse).

This is **not** the chat-client contract (`metadata.error` / SSE
`type: error` — see [generation-errors](api/generation-errors.md)).
Cancellation is not an error: it is `Message.stopped` / SSE `stopped`.
Endpoint circuit state (`EndpointHealth`) is not written here.

New documents omit keys that would be empty or a copy of another field. A
missing key means “not applicable”, not “unknown”. `kind` **is always set.**

## How a row is written

```text
producer  →  ErrorLogger.log_error()  →  in-memory buffer
                                          ├─ flush every 5s or 100 rows
                                          └─ ErrorLog.bulk_create() → Mongo error_logs

                 if kind is agentic ──► record_error_kind()  (no-op without Langfuse keys)
```

1. Request routers set contextvars (`user_id`, `conversation_id`,
   `message_id`) so every later `log_error` can attach chat identity
   without threading IDs through the graph.
2. Call sites either call `ErrorLogger.log_error` / `log_error_sync`
   directly, or `persist_policy_event` (a thin wrapper that sets
   `kind` from a policy name).
3. `ErrorLog.collapse_mirrored_fields` drops duplicated payload keys and
   flattens legacy `kind="policy"` + `policy="timeout"` into `kind="timeout"`.
4. Frontend rows skip the buffer: `POST /log-error` constructs `ErrorLog`
   and `save()`s immediately.

`ErrorLogger` is a process singleton from `get_error_logger()`. Logging
failures never raise to the caller.

## Classes

### Backend (this repo)


| Class / function | Module | Role |
| ---------------- | ------ | ---- |
| `ErrorLog` | `src/database/models/error_log.py` | Mongo document (`MongoModel`, collection `error_logs`) |
| `ErrorLogger` | `src/utils/error_logger.py` | Serialize, redact, buffer, bulk-insert |
| `PolicyEvent` | `src/utils/error_logger.py` | Stand-in exception when there is no real Python failure (retry, skipped MCP, …) |
| `Component` | `src/utils/error_logger.py` | Legacy RAG labels: `LLM`, `LLM_FALLBACK`, `RETRIEVAL`, `RETRIEVAL_FALLBACK`, `RE-RANKER`, `RE-RANKER_FALLBACK`, `ROUTER`, `MCP_TOOL` |
| `PipelineStage` | `src/utils/error_logger.py` | Legacy RAG stages: `generation`, `retrieval`, `re-querying`, `router`, `tool_execution`, `hallucination` |
| `persist_policy_event` | `src/utils/error_logger.py` | Agentic / MCP-load helper: `kind` = policy name |
| `get_error_logger` | `src/utils/error_logger.py` | Singleton accessor |
| `set_*_context` | `src/utils/error_logger.py` | Request-scoped `user_id` / `conversation_id` / `message_id` |
| `FrontendErrorLogRequest` | `src/schemas/error_log.py` | Body for `POST /log-error` |
| `ErrorLoggingInterceptor` | `src/services/agents/core/interceptors.py` | MCP tool interceptor (`ToolCallInterceptor` protocol) |
| `ToolReturnedError` | `src/services/agents/core/interceptors.py` | Exception stand-in when the tool returned `isError` / structured / 401 text |
| `classify_mcp_tool_result` | `src/services/agents/core/interceptors.py` | Detect returned MCP failures (`is_error` / `structured` / `text`) |
| `_graph_on_policy` | `src/services/agents/core/runner.py` | Callback passed into `AgentGraph.compile(on_policy=…)` |
| `_AgenticBudgetTimeout` | `src/services/agents/core/runner.py` | Whole-run `AGENTIC_TIMEOUT` (`kind=run_timeout`); does not open the LLM circuit |
| `_NodeKindCallbackHandler` | `src/utils/langfuse_helper.py` | Langfuse `CallbackHandler` that caches node spans after they pop |
| `record_error_kind` | `src/utils/langfuse_helper.py` | Optional Langfuse EVENT on the node span |
| `langfuse_context` / `get_callbacks` | `src/utils/langfuse_helper.py` | Wrap LangGraph invoke; no-op when keys are missing |


`Component.MCP_TOOL` and `PipelineStage.HALLUCINATION` exist on the enums
but have no current writers.

### Graph package (`eve-esa-agents`, import `agents.graphs.base`)

Resolved by `src/services/agents/graphs_bundle.py`. The graph must not
import `src.*`; it only calls `on_policy`.


| Class / method | Role |
| -------------- | ---- |
| `AgentGraph` | Base graph: `compile`, `timed_node`, `error_handler`, `retry_policy`, `timeout_policy` |
| `AgentGraph.retry_policy()` | LangGraph `RetryPolicy(max_attempts=3, retry_on=retry_on_transient)` |
| `AgentGraph.timeout_policy()` | LangGraph `TimeoutPolicy(run_timeout=…, idle_timeout=…)` |
| `AgentGraph.timed_node()` | Wraps a node; on `node_attempt > 1` emits `policy="retry"` |
| `AgentGraph.error_handler()` | Graph-level handler: emits `timeout` or `error_handler`, then `fallback` and `Command(goto="agent_fallback")` |
| `emit_on_policy` | Invokes the backend callback; never raises into the graph |
| `LatencyInterceptor` | MCP latency logs only (not `error_logs`) |
| `retry_on_transient` | Retry `ConnectionError`, timeouts, LangGraph default HTTP failures — not `ValueError` |


### LangGraph (library)

`RetryPolicy`, `TimeoutPolicy`, `NodeTimeoutError`, `NodeError` from
`langgraph.types` / `langgraph.errors`. `NodeTimeoutError` is **not** a
stdlib `TimeoutError`; the runner maps it with `_is_node_timeout_error`.

## Typology (`kind`)

`kind` is the only discriminator. Switch the backoffice UI on this field.


| `kind` | What happened | Typical location | Also Langfuse? |
| ------ | ------------- | ---------------- | -------------- |
| `timeout` | Node idle/run timeout (`NodeTimeoutError`) | `node` + `graph` | yes |
| `run_timeout` | Whole-run budget (`AGENTIC_TIMEOUT` / `_AgenticBudgetTimeout`) | `source=runner`, `graph` | yes |
| `retry` | LangGraph retried a node (`timed_node`, attempt > 1) | `node`, `graph`, `error.attempt` | yes |
| `error_handler` | Uncaught exception in a node, leftover runner catch, or agentic router | `node` and/or `source=router`/`runner` | yes |
| `fallback` | In-graph fallback LLM after primary failed | `node`, `graph` | yes |
| `mcp_load` | MCP server skipped or failed to list tools | `source=mcp_load`, `error.server` | no |
| `tool_error` | MCP tool call failed | `node=tools`, `source=mcp` | yes |
| `rag` | Classic retrieval / generation pipeline | `component`, `pipeline_stage` | no |
| `frontend` | Browser `POST /log-error` | `source=frontend` | no |


Old rows that still have `kind: "policy"` plus a separate `policy`
column are read as `kind` = that policy name.

## How graphs emit and how we intercept

The graph package (`eve-esa-agents`) **never writes Mongo**. It has no
`src.*` imports. Failures are either (a) observed through a callback the
backend injects at compile time, or (b) intercepted below the graph,
inside the MCP client, before LangGraph sees an exception.

Three layers, from inside the node outward:

```text
MCP tool.ainvoke
  └─ ErrorLoggingInterceptor          ← layer B (tool_error). Most MCP
       LatencyInterceptor                failures never leave the tools node.
       ArtifactInterceptor
            │
            ▼
tools / agent / agent_fallback node body
  └─ AgentGraph.timed_node            ← observes retries (not an except)
            │  exception escapes the node
            ▼
LangGraph RetryPolicy / TimeoutPolicy ← layer A (retry, timeout)
  └─ AgentGraph.error_handler         ← layer A (timeout | error_handler | fallback)
            │  handler re-raises, or no handler
            ▼
runner astream except                 ← layer C (source=runner leftover)
```

### Compile-time wiring

`runner._build_react_graph` compiles with the backend callback and the
node budgets:

```text
agent.compile(
    llm=…, tools=…, fallback_llm=…,
    llm_idle_timeout=MODEL_TIMEOUT,   # TimeoutPolicy idle (resets on chunks)
    llm_run_timeout=AGENTIC_TIMEOUT,  # TimeoutPolicy wall-clock per attempt
    on_policy=_graph_on_policy(graph.name),
)
```

`ReactAgent.compile` / `SimpleChatAgent.compile` then attach LangGraph
kwargs **per node** via `add_instrumented_node` (drops kwargs the installed
LangGraph does not accept):


| Node | `timed_node` | `RetryPolicy` | `TimeoutPolicy` | `error_handler` |
| ---- | ------------ | ------------- | --------------- | --------------- |
| `agent` | yes | yes (max 3, `retry_on_transient`) | idle + run | yes → may `goto agent_fallback` |
| `tools` (react only) | yes | yes | **no** | yes, but see layer B |
| `agent_fallback` | yes | **no** | idle + run | yes, then **re-raises** (`node == agent_fallback`, no loop) |


`retry_on_transient` retries `ConnectionError`, stdlib `TimeoutError`,
`NodeTimeoutError`, and LangGraph’s default HTTP failures — not
`ValueError`.

### The callback (graph → Mongo)

The graph calls `emit_on_policy(on_policy, …)`. That helper awaits the
callback and **swallows callback errors** so logging cannot crash the
graph.

Backend side (`runner._graph_on_policy`):

```text
on_policy(node, policy, attempt, error, extra, description)
    → persist_policy_event(policy=policy, node=node, graph=graph_name, …)
        → ErrorLogger.log_error(kind=policy, …)
```

`timed_node` is **not** an interceptor. It does not `except`. On every
LangGraph re-invoke it reads `runtime.execution_info.node_attempt`; if
that is `> 1` it emits `policy="retry"` and then runs the node function.

`error_handler` **is** the interceptor for exceptions that left the node
body. LangGraph invokes it with a `NodeError` (`error.node`,
`error.error`). The handler:

1. Sets `policy = "timeout"` if the cause is `NodeTimeoutError`, else
   `"error_handler"`.
2. Emits that policy via `on_policy`.
3. If `fallback_llm` is set and the failed node is not `agent_fallback`:
   emits `policy="fallback"` and returns `Command(goto="agent_fallback")`.
   The exception is **swallowed by the graph**; the run continues.
4. Otherwise re-raises. The exception leaves `astream` and layer C runs.

### Layer A — LangGraph policies (exceptions that escape a node)

Typical `agent` timeout (primary LLM stalled):

1. `TimeoutPolicy` wraps **one attempt**. Idle timeout resets on every
   streamed chunk (first-token budget). Run timeout is a hard wall clock
   and does not reset. Either one raises `NodeTimeoutError` (this is
   **not** a stdlib `TimeoutError`).
2. `RetryPolicy` sees a transient failure and re-invokes `agent`
   (attempts 2 and 3). Each re-invoke goes through `timed_node`, which
   emits `kind=retry` with `error.attempt`.
3. After attempts are exhausted, LangGraph calls `error_handler`.
4. Handler emits `kind=timeout`, then `kind=fallback`, then jumps to
   `agent_fallback`.
5. `agent_fallback` is a single LLM call on the fallback model. If
   **it** times out, the same handler emits `timeout` (or `error_handler`)
   and re-raises — no second fallback.

A programmer error (`ValueError`, bad prompt, …) skips retry (not
transient), goes straight to `error_handler` → `kind=error_handler` →
fallback if available.

### Layer B — MCP interceptor (inside `tools`, before LangGraph)

Most tool failures **never become graph exceptions**. That is why
`RetryPolicy` / `error_handler` on `tools` almost never fire, and why
`tool_error` is not produced by `on_policy`.

`tool_loader._discover_mcp_tools_uncached` builds:

```text
MultiServerMCPClient(
    connections,
    tool_interceptors=[
        LatencyInterceptor(),          # log duration only
        ErrorLoggingInterceptor(),     # writes error_logs
        ArtifactInterceptor(),
    ],
)
```

`ErrorLoggingInterceptor` implements the langchain-mcp-adapters
`ToolCallInterceptor` protocol: `async (request, handler) → result`.


| What the tool does | Interceptor | Then `make_tools_node` | Graph policies |
| ------------------ | ----------- | ---------------------- | -------------- |
| Raises | log `tool_error` `signal=exception`, **re-raise** | `except` → `ToolMessage("Tool error: …")` | never see it |
| Returns `isError` / JSON-RPC error | log `signal=is_error`, **return result** | success path, ToolMessage with payload | never see it |
| Returns `{error: …}` | log `signal=structured`, return | success | never see it |
| Returns 401-ish text | log `signal=text`, return | success | never see it |
| Success | no log | success | n/a |


The ReAct loop then continues (`tools` → `agent`) with the failure as
ordinary tool output. A Wiley 401 must not retry the tools node or jump
to `agent_fallback`.

`error_handler` on `tools` would only run if something escaped
`make_tools_node`’s per-tool `try/except` (almost nothing does).

MCP **load** (missing URL, `get_tools` failed) is not a graph event at
all: `tool_loader` calls `persist_policy_event(policy="mcp_load")`
directly. `kind=mcp_load` never hits Langfuse.

`error.signal` for `tool_error`:


| Value | Meaning |
| ----- | ------- |
| `exception` | The tool handler raised |
| `is_error` | MCP `CallToolResult.isError` or JSON-RPC error |
| `structured` | Tool returned `{error: …}` |
| `text` | 401-style plain text (`Error 401`, `unauthorized`, …) |


### Layer C — runner leftover (escaped the compiled graph)

If `error_handler` re-raises, or a budget outside the node fires,
`graph.astream(...)` raises into `generate_answer_agentic_helper` /
`generate_answer_agentic_stream_helper`. Those `except` blocks call
`persist_policy_event` with `source=runner`. `_policy_for_uncaught`:

- `_AgenticBudgetTimeout` or stdlib `TimeoutError` → `run_timeout`
- `NodeTimeoutError` → `timeout`
- anything else → `error_handler`

`_AgenticBudgetTimeout` is **not** LangGraph. Streaming checks elapsed
time until the first **final-answer** token against `AGENTIC_TIMEOUT`
(tool time counts). It must not open the LLM circuit: slow MCP would
otherwise park a healthy endpoint.

If `agent_fallback` already produced an answer, the runner may persist
the leftover **and** keep the fallback answer. That leftover can
**duplicate** an in-graph `timeout` already written by `error_handler`.


| Function | When | Typical `kind` |
| -------- | ---- | -------------- |
| helper leftover after fallback | graph raised but fallback already answered | `timeout` / `error_handler` |
| helper outer `except` | generation failed with no usable answer | same |
| stream leftover after fallback | same, streaming | same |
| streaming `except TimeoutError` | run/node timeout escaped the graph | `timeout` / `run_timeout` |
| streaming `except Exception` | any other escaped failure | `error_handler` (or timeout mapping) |


### Agentic HTTP router

Still outside the graph: `create_agentic_message_stream` in
`src/routers/message.py` logs `kind=error_handler`, `source=router` on
HTTP/uncaught exceptions around stream setup (not node execution).
Contextvars are set at the start of the handler.

Classic `create_message_stream` uses `Component.ROUTER` /
`PipelineStage.ROUTER` with no explicit `kind`, so the model infers
`kind=rag`.

### MCP load (not a graph intercept)


| Producer | File | Function | `kind` | Notes |
| -------- | ---- | -------- | ------ | ----- |
| Missing MCP URL | `src/services/mcp/tool_loader.py` | `_discover_mcp_tools_uncached` | `mcp_load` | `PolicyEvent` |
| `get_tools` failed | same | `_load_one` | `mcp_load` | real exception |
| Tool raised / returned failure | `src/services/agents/core/interceptors.py` | `ErrorLoggingInterceptor` | `tool_error` | layer B above |


### Classic RAG

All of these go through `get_error_logger().log_error` with `Component` +
`PipelineStage`. `ErrorLog` infers `kind=rag` when those are set and
`kind` was omitted.


| File | Function / area | `component` | `pipeline_stage` |
| ---- | --------------- | ----------- | ---------------- |
| `src/services/generate_answer.py` | DeepInfra / SiliconFlow rerank | `RE-RANKER` | `retrieval` |
| same | MCP auth / retrieval in classic path | `RETRIEVAL` | `retrieval` |
| same | `should_use_rag` main / fallback | `LLM` / `LLM_FALLBACK` | `re-querying` |
| same | `get_rag_context` vector search | `RETRIEVAL` | `retrieval` |
| same | `setup_rag_and_context` (sync + stream) | `RETRIEVAL` | `retrieval` |
| same | scraping-dog fallback | `RETRIEVAL` | `retrieval` |
| same | LangGraph invoke / stream timeout or fail | `LLM` | `generation` |
| same | outer generation / streaming catch | `LLM` | `generation` |
| `src/core/vector_store_manager.py` | query embedding main / fallback | `RETRIEVAL` / `RETRIEVAL_FALLBACK` | `retrieval` |
| `src/routers/message.py` | `create_message_stream` HTTP/uncaught | `ROUTER` | `router` |


### Frontend

`src/routers/error_log.py` — `POST /log-error`. Authenticated. Writes
`kind=frontend`, `source=frontend`, `component` from the body (default
`FRONTEND`), `pipeline_stage=CLIENT_ERROR`. Direct `ErrorLog.save()`,
not the batch logger.

## Document schema

### Always present


| Field | Meaning |
| ----- | ------- |
| `_id` | Mongo id |
| `timestamp` | When the row was written (UTC) |
| `kind` | Discriminator — switch the backoffice UI on this |
| `description` | Human-readable summary (secrets redacted, max 500 chars) |


### Context (when the failure happened inside a chat request)


| Field | Meaning |
| ----- | ------- |
| `user_id` | Authenticated user |
| `conversation_id` | Conversation |
| `message_id` | Assistant message shell |


Absent when the logger ran outside a request (for example MCP load at
startup) or when request context was not attached.

### Location


| Field | Set on | Meaning |
| ----- | ------ | ------- |
| `node` | In-graph agentic failures | LangGraph node: `agent`, `agent_fallback`, `tools`, … |
| `graph` | Agentic | Graph name: `react`, `simple`, … |
| `source` | Off-graph / no node | `runner`, `mcp_load`, `router`, `mcp`, `frontend` |
| `logger_name` | RAG (and some agentic) | Python module. Omitted when it equals `source` |


### `error` object (optional)

Omitted entirely when there is nothing to add beyond `description`.


| Key | Meaning |
| --- | ------- |
| `type` | Exception class (`TimeoutError`, `HTTPException`, `ToolReturnedError`, …). Not stored for the synthetic `PolicyEvent` stand-in |
| `message` | Exception text, **only if it differs from** `description` |
| `args` | Exception args, only if they are not a copy of the message |
| `attempt` | Retry count (when `kind=retry`) |
| `server` | MCP server name (`kind=mcp_load` or `tool_error`) |
| `tool` | MCP tool name (`kind=tool_error`) |
| `signal` | How the tool failure was detected (`kind=tool_error`) |
| `stack` / `url` / `user_agent` / `metadata` | Frontend only |


### RAG-only columns (`kind=rag`)


| Field | Values |
| ----- | ------ |
| `component` | `LLM`, `LLM_FALLBACK`, `RETRIEVAL`, `RETRIEVAL_FALLBACK`, `RE-RANKER`, `RE-RANKER_FALLBACK`, `ROUTER`, `MCP_TOOL` |
| `pipeline_stage` | `generation`, `retrieval`, `re-querying`, `router`, `tool_execution`, `hallucination` |


Agentic rows do **not** set these. Use `node` / `kind` instead.

### Not written on new rows


| Field | Why |
| ----- | --- |
| `policy` | Flattened into `kind` |
| `error_type` | Duplicate of `error.type` |
| `component` / `pipeline_stage` | Agentic: use `node` / `kind`. Still written for `rag` and `frontend` |


Documents written before this taxonomy may still contain those copies.

## Langfuse overlay

Enabled only when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are
set (`is_langfuse_enabled`). `LANGFUSE_BASE_URL` selects the instance.

`ErrorLogger.log_error` calls `record_error_kind` for
`timeout`, `run_timeout`, `retry`, `error_handler`, `fallback`,
`tool_error`. The helper nests a Langfuse EVENT under the named node
span (`agent`, `agent_fallback`, `tools`, …). `mcp_load`, `rag`, and
`frontend` stay Mongo-only.

Without keys the helper returns immediately; Mongo still receives the
row. Hosted can turn Langfuse on later by adding URL + keys — no
producer changes.

## Example

Timeout (no extra exception payload):

```json
{
  "kind": "timeout",
  "description": "Policy 'timeout' on node 'agent'",
  "node": "agent",
  "graph": "react",
  "user_id": "…",
  "conversation_id": "…",
  "message_id": "…"
}
```

## What is not in `error_logs`

- User cancel / stop (`asyncio.CancelledError` → `stopped`).
- Circuit-open / cooldown (`EndpointHealth.record_failure` in
  `src/core/llm_manager.py`). The leftover timeout/error_handler row
  already records the failure; circuit state is on
  `metadata.endpoint.circuit_open`.
- Chat-visible `metadata.error` / SSE `type: error` (separate contract).
- Successful retries of a later node attempt (only `kind=retry` for the
  re-entry is logged; the successful attempt is a normal trace step).
