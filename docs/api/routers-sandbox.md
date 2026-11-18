# Conversation files and sandbox API

Two resources under a conversation. `files` is what the conversation saved in object storage. `sandbox` is the live provider session, and `sandbox/files` is its tree. See [Sandbox](../sandbox.md) for how the two stay in step.

The routes answer 404 unless `FEATURE_SANDBOX_API` is `true`, and they are left out of the schema while it is off. Today only GET exists. A conversation that is missing or owned by another user answers 404 `Conversation not found`.

| Route | Returns |
| --- | --- |
| `GET /conversations/{conversation_id}/files` | `FileListing` of saved files |
| `GET /conversations/{conversation_id}/files/{path}` | The bytes of one saved file |
| `GET /conversations/{conversation_id}/sandbox` | `SandboxRead` |
| `GET /conversations/{conversation_id}/sandbox/files` | `FileListing` of the live session |

None of them starts a sandbox, takes the pool lock, or moves the idle deadline.

## Saved files

`GET /conversations/{conversation_id}/files?prefix=&limit=`

::: routers.sandbox.list_conversation_files
    options:
      show_root_heading: false
      show_source: false

Paths are relative to `users/{user_id}/conversations/{conversation_id}/`: `workspace/...`, `artifacts/{artifact_id}/{filename}`, `large_tool_results/...`, and `workspace_state`. `prefix` keeps paths that start with it. `limit` is 1 to 1000, default 1000. When there are more objects, `truncated` is true and the newest `limit` are returned, sorted by path.

`GET /conversations/{conversation_id}/files/{path}` streams one object. It uses the same rule as `GET /conversations/{conversation_id}/workspace/{path}`: only the tool allowlist keeps its type and renders inline. Anything else is `application/octet-stream` with `attachment`. A path that is empty, absolute, or has a `.` or `..` segment answers 404.

```python
resp = requests.get(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}/files",
    params={"prefix": "workspace/"},
    headers=headers,
    timeout=30,
)
for item in resp.json()["items"]:
    print(item["path"], item["size"])
```

## Sandbox

`GET /conversations/{conversation_id}/sandbox`

::: routers.sandbox.get_conversation_sandbox
    options:
      show_root_heading: false
      show_source: false

The answer is read from the pool map, not from the provider. A sandbox the provider already ended still reads `idle` until the reaper or the next turn drops the record.

| `status` | Meaning |
| --- | --- |
| `disabled` | `SANDBOX_PROVIDER` is unset or `none` |
| `none` | No sandbox for this conversation |
| `idle` | A sandbox, and no turn is using it |
| `busy` | A turn is using it (`inflight` > 0) |

`idle_deadline` is when the reaper stops it. `created_at` and `expires_at` (`created_at` plus `max_lifetime_s`, the hard cap) are set for AgentCore only.

## Live sandbox files

`GET /conversations/{conversation_id}/sandbox/files?prefix=&limit=`

::: routers.sandbox.list_sandbox_files
    options:
      show_root_heading: false
      show_source: false

The route attaches to the session, runs one `find` over `workspace`, `skills`, `artifacts`, and `large_tool_results`, and detaches. Paths are relative to the session root. `limit` is 1 to 2000, default 2000.

| Status | When |
| --- | --- |
| 404 `Sandbox not found` | No record for this conversation |
| 410 `Sandbox gone` | The provider reports the session ended. The record stays |
| 502 | Attach or the listing failed. The detail names the error class |

## Later

These paths are kept for write methods. None of them exists yet.

| Method and path | Intent |
| --- | --- |
| `POST /conversations/{id}/sandbox` | Start or warm the sandbox |
| `DELETE /conversations/{id}/sandbox` | Discard it (`discard_conversation_sandbox`) |
| `GET`, `PUT`, `DELETE /conversations/{id}/sandbox/files/{path}` | Read, write, remove a live file |
| `PUT`, `DELETE /conversations/{id}/files/{path}` | Write, remove a saved file |
