# Bug reports

Users report a bug from the chat. The report carries what they wrote, the browser context
(session replay, last trace id, conversation and message ids, build, viewport, recent console
errors). There is no screenshot: the HyperDX session replay linked by `replay_url` shows what
the user saw. Routes live in `routers.bug_report`, logic in `src.services.bug_reports`.

## File a report

`POST /bug-reports`, authenticated, `multipart/form-data`:

| Field | Type | Rules |
|---|---|---|
| `description` | text | 1 to 4000 characters, not blank |
| `context` | JSON text | object below, required (`{}` is valid) |

Any other part is ignored. A `screenshot` file part, sent by frontends older than
2026-09-24, is dropped unread and the report is stored as if it were absent.

`context`, every key optional, unknown keys dropped:

```json
{
  "session_id": "rum session id",
  "replay_url": "HyperDX replay deep link: /sessions?sid=...&sfrom=...&sto=...&ts=...",
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

- `201` `{"id", "created_at"}`.
- `401` without a valid credential.
- `404` when `context.conversation_id` is missing or not the caller's. Nothing is stored.
- `422` blank or too long description, missing context, invalid context JSON or field.
- `429` `{"detail": {"code": "bug_report_rate_limited", ...}}` with `Retry-After: 3600` after
  `BUG_REPORT_MAX_PER_HOUR` (default 5) reports in the rolling hour. Counted on the
  `bug_reports` collection, no Redis; `<=0` disables it. The whole limit is behind
  `FEATURE_BUG_REPORT_RATE_LIMIT` (default on): `false` or `0` means no limit at all.

```bash
curl -X POST "$BASE_URL/bug-reports" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F 'description=The answer stopped halfway' \
  -F 'context={"conversation_id":"...","path":"/chat/..."}'
```

## Read a report back

`GET /bug-reports/{id}` returns the whole stored report to its author: description, context,
conversation snapshot and `request_trace_id`. Anyone else and an unknown id answer `404`.

## Storage

- Collection `bug_reports`: `user_id`, `description`, `context` (every field above, `null` when
  not sent), `conversation` (server side snapshot or `null`), `request_trace_id`
  (trace of the POST, `null` with telemetry off), `timestamp`. Index
  `bug_reports_by_user_time` backs the rate limit.
- Description and every context string are redacted with `src/utils/redaction.py` before
  they are stored: bearer tokens, JWTs, `eve_` keys, credential query parameters and email
  addresses become placeholders.
- `replay_url` goes through the same redaction and is otherwise stored as sent: its `sid`,
  `sfrom`, `sto` and `ts` parameters carry no secret and are left untouched.
- Reports filed before 2026-09-24 may still hold a `screenshot` subdocument and objects under
  `bug-reports/` in the artifacts bucket. Nothing reads them any more.

## Log event

Each stored report emits `bug_report.created` at WARNING on logger `eve.bug_report`, inside
the request span, so it carries the request's trace and span id when telemetry is on. With
telemetry off it is a plain WARNING line. Attributes (absent values left out):

`event.name`, `eve.bug_report.id`, `rum.sessionId`, `eve.replay_url`,
`gen_ai.conversation.id`, `eve.message_id`, `user.id`, `deployment.environment.name`,
`eve.bug_report.messages`, `eve.bug_report.truncated`. `eve.replay_url` is the deep link as
stored.

The description is never in the event. The request span also gets `eve.bug_report.id` and
`user.id`.
