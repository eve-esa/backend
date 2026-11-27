# Storage quota

Each user has one byte counter, `storage_bytes_used`, for the current objects under `users/{user_id}/`. Noncurrent versions are not counted: the user cannot see them, and the bucket lifecycle expires them. The counter changes only by an atomic `$inc`, on the write and on the delete. It is never written by saving the user document.

The unit of `FREE_STORAGE_GB`, `PRO_STORAGE_GB`, `PRO_PLUS_STORAGE_GB` and `ULTRA_STORAGE_GB` is GiB (1024^3), the unit AWS uses for S3 billing. The plan names are the same ones as the token budget, including the aliases. A plan with no variable, or with `0`, has no cap. The `unlimited` plan never has a cap. A user who is above a cap after a downgrade keeps their files; only a new write is refused.

`FEATURE_STORAGE_QUOTA` defaults to off. While it is off, bytes are still counted and nothing is refused. Turn it on only after `python -m src.commands.reconcile_storage_usage --apply` has filled the counter for that environment. The command is a dry run without `--apply`. It adds the difference between the listing and the counter it read. It does not replace the counter, so a write during the listing is not lost. A second run converges.

## `POST /artifacts`

A file that would pass the cap is **403**, with no `Retry-After`. Freeing space is the user's action, not a wait. 507 is the WebDAV code for a full quota and is not used here: a 5xx would look like a server fault.

```json
{
  "detail": {
    "code": "storage_quota_exceeded",
    "message": "Storage is full. Delete files or conversations to free space.",
    "used_bytes": 21474836480,
    "max_bytes": 21474836480,
    "needed_bytes": 5242880
  }
}
```

`DELETE /artifacts/{id}` gives those bytes back.

## Inside an agent turn

The turn keeps running.

* A file from an MCP tool is not stored. The tool call fails. The model and the chip see only `Tool error: Storage is full. Delete files or conversations to free space.` The tool's own output is dropped.
* A sandbox file that does not fit is not synced. It stays on the sandbox disk. The next turn reports it.
* A large tool result is counted and is never refused. Refusing it would put the whole result back into the prompt.

The legacy JSON stream sends one frame, once per turn:

```json
{"type": "notice", "code": "storage_quota_exceeded", "used_bytes": 21474836480, "max_bytes": 21474836480}
```

The AG-UI adapter maps that frame to a `CUSTOM` event named `eve.notice`. `RUN_ERROR` is not used, because it ends the run. The value is:

```json
{
  "code": "storage_quota_exceeded",
  "message": "Storage is full. Delete files or conversations to free space.",
  "usedBytes": 21474836480,
  "maxBytes": 21474836480
}
```

The `final` frame and the sync `generate-agentic` JSON include `"notices": ["storage_quota_exceeded"]` when this happened. The key is omitted when it did not.

A workspace sync runs after the answer, so that stream is already closed. The next turn sends the same notice before the first token when the user is at the cap, or when a write was refused after the stream closed.

## `GET /users/me/storage-usage`

```json
{
  "unlimited": false,
  "rate_limit_group": "free",
  "used_bytes": 1048576,
  "max_bytes": 21474836480,
  "remaining_bytes": 21473787904,
  "used_ratio": 0.0,
  "used_gb": 0.0,
  "max_gb": 20.0
}
```

`used_gb` and `max_gb` are GiB rounded to 2 decimals. `used_bytes` is never negative. `unlimited` is true when the plan has no cap, and the cap fields are then null.

## Conversation delete

Deleting a conversation removes the whole prefix `users/{user_id}/conversations/{conversation_id}/`. An artifact that another of the owner's conversations still shows is moved, object and row, into that conversation first. The bytes of every removed object are subtracted from the counter.
