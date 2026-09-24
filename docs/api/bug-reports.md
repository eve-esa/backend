# Bug reports

Users report a bug from the chat. The report carries what they wrote, the browser context
(session replay, last trace id, conversation and message ids, build, viewport, recent console
errors) and an optional screenshot. Routes live in `routers.bug_report`, logic in
`src.services.bug_reports`.

## File a report

`POST /bug-reports`, authenticated, `multipart/form-data`:

| Field | Type | Rules |
|---|---|---|
| `description` | text | 1 to 4000 characters, not blank |
| `context` | JSON text | object below, required (`{}` is valid) |
| `screenshot` | file, optional | PNG or JPEG, at most 1 MB (`BUG_REPORT_SCREENSHOT_MAX_BYTES`) |

`context`, every key optional, unknown keys dropped:

```json
{
  "session_id": "rum session id",
  "replay_url": "replay deep link",
  "trace_id": "last trace id the page saw",
  "conversation_id": "...",
  "message_id": "...",
  "app_version": "v0.1.1",
  "app_commit": "8ea24e5",
  "environment": "dev",
  "user_agent": "...",
  "viewport": {"width": 1440, "height": 900},
  "path": "/chat/...",
  "console_errors": ["at most 50 entries, each clipped to 2000 chars"],
  "privacy_mode": "off | mask | on_demand | clear"
}
```

Answers:

- `201` `{"id", "created_at", "screenshot": bool}`. `screenshot` is `false` when none was sent,
  or when storing it failed: the report is kept anyway.
- `401` without a valid credential.
- `413` screenshot above the cap. `415` screenshot that is not PNG or JPEG.
- `422` blank or too long description, missing context, invalid context JSON or field.
- `429` `{"detail": {"code": "bug_report_rate_limited", ...}}` with `Retry-After: 3600` after
  `BUG_REPORT_MAX_PER_HOUR` (default 5) reports in the rolling hour. Counted on the
  `bug_reports` collection, no Redis; `<=0` disables it. The whole limit is behind
  `FEATURE_BUG_REPORT_RATE_LIMIT` (default on): `false` or `0` means no limit at all.

```bash
curl -X POST "$BASE_URL/bug-reports" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F 'description=The answer stopped halfway' \
  -F 'context={"conversation_id":"...","path":"/chat/..."}' \
  -F 'screenshot=@shot.png;type=image/png'
```

## Read the screenshot back

`GET /bug-reports/{id}/screenshot` streams the image inline to the report's author. Anyone
else, an unknown id, a report without screenshot and a missing object all answer `404`.

## Storage

- Collection `bug_reports`: `user_id`, `description`, `context` (every field above, `null` when
  not sent), `screenshot` (`key`, `content_type`, `size_bytes`, or `null`), `request_trace_id`
  (trace of the POST, `null` with telemetry off), `timestamp`. Index
  `bug_reports_by_user_time` backs the rate limit.
- Description and every context string are redacted with `src/utils/redaction.py` before
  they are stored: bearer tokens, JWTs, `eve_` keys, credential query parameters and email
  addresses become placeholders.
- The screenshot type is sniffed from its bytes with `sniff_artifact_type`, never taken from
  the declared type. It is stored in the artifacts bucket at
  `bug-reports/{user_id}/{id}.{png|jpeg}`, outside the `users/` artifact tree, and is
  never logged.

## Log event

Each stored report emits `bug_report.created` at WARNING on logger `eve.bug_report`, inside
the request span, so it carries the request's trace and span id when telemetry is on. With
telemetry off it is a plain WARNING line. Attributes (absent values left out):

`event.name`, `eve.bug_report.id`, `rum.sessionId`, `eve.replay_url`,
`gen_ai.conversation.id`, `eve.message_id`, `user.id`, `deployment.environment.name`.

The description is never in the event. The request span also gets `eve.bug_report.id` and
`user.id`.
