## Sandbox

A conversation keeps one sandbox. A message does not create a new one and does not destroy it. The next message reuses that sandbox and restarts the idle timer, so a file written in one turn is still there on the next turn. The sandbox goes away when it has been idle, when it is replaced, or when the conversation is deleted.

The provider session is `src/services/agents/core/sandbox.py`. The map shared by the workers is `src/services/agents/core/sandbox_pool.py`. `src/services/agents/core/runner.py` acquires the sandbox for a turn and releases it afterwards. Deleting a conversation calls `discard_conversation_sandbox`. A reaper starts with the server and stops on shutdown. Shutdown does not destroy sandboxes.

A sandbox starts only when the graph's `compile` function takes a `backend` argument. Today that is the deep agent (`"agent": "deep"` on [agentic generation](api/routers-message.md#agentic-generation)). `SANDBOX_PROVIDER` unset or `none` starts nothing. A turn with no conversation id still uses `open_sandbox` / `close_sandbox`, and that sandbox is destroyed at the end of the turn, including when the turn fails. A streaming turn always has a conversation id and uses the pool.

### Providers

Settings are read when a turn starts, not at import. Every name this service owns starts with `SANDBOX_`. A provider SDK that already has its own names keeps those names.

| Provider | What it is | Settings |
| --- | --- | --- |
| `agentcore` | Bedrock AgentCore code interpreter, through `langchain-agentcore-codeinterpreter` | `SANDBOX_AGENTCORE_REGION` (required). `SANDBOX_AGENTCORE_IDENTIFIER` (default `aws.codeinterpreter.v1`) |
| `opensandbox` | `opensandbox` `SandboxSync` plus `langchain-sandbox-opensandbox` | `SANDBOX_OPENSANDBOX_DOMAIN` (required). `SANDBOX_OPENSANDBOX_PROTOCOL` (`http` or `https`, default `http`). `SANDBOX_OPENSANDBOX_API_KEY` (optional). `SANDBOX_OPENSANDBOX_IMAGE` (default `python:3.12`) |

AgentCore `sessionTimeoutSeconds` is an absolute cap. Activity does not extend it, and the API cannot renew it. The cap is 28,800 seconds ([`StartCodeInterpreterSession`](https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_StartCodeInterpreterSession.html)). `SANDBOX_MAX_LIFETIME_S` defaults to 3600 and is clamped to that cap. A session is reusable only while the time left on the cap is greater than `AGENTIC_TIMEOUT` plus 60 seconds. Status `READY` can be attached. The sandbox is gone when the status is `TERMINATED`, or when `get_session` raises `ResourceNotFoundException`, a `ValidationException` whose message says `is not active`, or a `ConflictException` whose message says `already terminated`. Any other `get_session` error, and any other status, fails the turn and leaves the record in place. A wrapper that did not create the session must not call `close()`: `close()` stops a session this process created. Attach stops the remote session through the client. The data-plane client read timeout is 300 seconds.

OpenSandbox `renew` sets the expiry to now plus the duration given. `destroy` kills the remote sandbox. `close` only drops the local HTTP client. A connect error whose `status_code` is 404 or 410 means the sandbox is gone. Anything else fails the turn and leaves the record.

### Idle timer and replacement

| Setting | Default | Meaning |
| --- | --- | --- |
| `SANDBOX_IDLE_TTL_S` | 600 | How long the sandbox stays after the last turn |
| `SANDBOX_MAX_LIFETIME_S` | 3600 | AgentCore hard cap. Not used for OpenSandbox |
| `SANDBOX_REAPER_INTERVAL_S` | 30 | How often a worker looks for idle sandboxes |
| `AGENTIC_TIMEOUT` | 120 | Turn cap. Added to the idle window while a turn is in flight |

While a turn is in flight the deadline is now plus the idle window plus `AGENTIC_TIMEOUT`, and an OpenSandbox expiry is renewed to that same window. The last turn to release sets the deadline back to now plus the idle window and renews OpenSandbox to the idle window. An earlier release of a concurrent turn does not pull the deadline in. The count of those turns is `inflight`.

A record is replaced only when the provider reports the sandbox gone, when the provider changed, or when an AgentCore session is inside the hard-cap margin and `inflight` is 0. A throttle, a timeout, or any other attach error keeps the record and fails the turn. A failed stop also keeps the record, so the reaper can try again. A failed save of a new sandbox destroys that new session. If the lock is lost after the new session was created, that session is destroyed and not saved.

A deadline that is already past, or missing, clears a stale `inflight` before the reuse decision. Otherwise a crashed turn would keep a session that is about to hit the AgentCore cap.

OpenSandbox records do not store `created_at` or `max_lifetime_s`. They stay reusable. Their expiry moves with `renew`.

### Redis

The map is in Redis when `REDIS_URL` is set, because the next message can land on another gunicorn worker. Without Redis the map is in this process only. Tests install a fresh in-process map so a `REDIS_URL` in CI cannot leak records between tests.

| Key | Type | Contents |
| --- | --- | --- |
| `sandbox:conv:{conversation_id}` | hash | `provider`, `sandbox_id`, `skills_digest`, `inflight`. AgentCore also stores `region`, `identifier`, `created_at`, `max_lifetime_s` |
| `sandbox:idle` | zset | Member is the conversation id. Score is the kill deadline as epoch seconds |
| `sandbox:lock:{conversation_id}` | string | `SET NX PX` for 180 seconds |

The lock covers create, attach, release, discard, and the reaper. It is not held while the turn runs, so two turns of one conversation share the sandbox. A waiter polls until the holder releases. There is no fixed wait. A dead holder stops refreshing and the key expires within 180 seconds. The reaper tries once and skips a lock that is held. The holder refreshes every 60 seconds. A Redis error on that refresh is retried. The lock is lost only when the refresh sees another token. Before every other write the holder asks Redis again and extends the key, so a write does not run on a lock that has already expired. The in-flight count and the deadline move in one script that also checks the token. A count that moved without its lease would never be released.

The reaper reads the deadline again under the lock. A turn that reattached between the scan and the lock is left alone.

### Skills and the graph backend

The skills digest is the sha256 of the uploaded `(path, bytes)` list, in upload order. The same digest skips the upload. A changed digest runs `rm -rf /skills` and uploads again. AgentCore stores `/skills/...` under the session working directory, because the process cannot create a real `/skills`. The shell rewrite points `/skills` at `{cwd}/skills`.

The backend passed to `compile` is a `CompositeBackend` whose default is the sandbox. `/large_tool_results/` and `/conversation_history/` stay on `StateBackend`, so those paths survive after the sandbox is destroyed. `/skills/` is not routed: the file tools and the shell both read the tree that was uploaded into the sandbox.

### Checking it

Use `"agent": "deep"`. Ask the model to write a file, then in a later message of the same conversation ask it to read that file. The backend log shows `sandbox.created` and then `sandbox.reused` with the same `sandbox_id`. Reading the file from the sandbox itself confirms the text. The model can also repeat a marker that was already in the first message. Discard the sandbox when the check is done.

Values already in the process environment are kept when `LOAD_DOTENV_OVERRIDE=false`. Otherwise a value in `.env` replaces them, including AWS credentials the provider SDK would otherwise take from the SSO chain.
