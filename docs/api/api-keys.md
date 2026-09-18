# API keys

Self-service `eve_` API keys for programmatic access, without copying a session token.
Routes live in `routers.user`, logic in `src.services.api_keys`.

## Concepts

- A key is a 32-byte random token, prefixed `eve_`, stored only as a SHA-256 hash. The raw
  token is returned exactly once, on create, and never again.
- Any valid credential (browser session or another `eve_` key) can create, list and revoke
  keys. There is no scope model yet: a key can do everything the session it stands in for
  can do.
- **Cap**: at most `API_KEY_MAX_ACTIVE_PER_USER` (default 10) active keys per user. Create
  past the cap answers 409.
- **Throttle**: at most `API_KEY_CREATE_MAX_PER_HOUR` (default 30) creations per user per
  rolling hour. Past that, 429.
- **Provenance**: a key created by another key records `created_by_key_id` and
  `created_via: "api_key"`. A key created from a browser session records
  `created_via: "oidc"` and no parent. Keys created before this feature shipped have both
  fields `null`.
- **Expiry clamp**: a key created by a parent key can never outlive its parent. If the
  requested expiry (or the default) is later than the parent's, it is silently clamped down.
- **Cascade revoke**: revoking a key also revokes every key it created, and their children,
  recursively. Revoke is idempotent: revoking an already-revoked key still answers 204.
- **Existence oracle closed**: revoking a key that does not exist and revoking another
  user's key both answer 404. There is no 403 that would tell a caller the key exists.

## Create

`POST /users/api-keys`

Body is optional; every field in it is optional. An empty body, or no body at all, takes
the defaults: a generated name and `API_KEY_DEFAULT_EXPIRES_IN_DAYS` (default 90).

```json
{
  "name": "CI pipeline",
  "expires_in_days": 30
}
```

- `name` (optional): stripped, max 100 chars, control and bidi-override characters
  rejected. Omitted: `Key created YYYY-MM-DD HH:MM UTC`.
- `expires_in_days` (optional): integer 1..3650. Omitted: the default (90 days).
  Explicit `null`: never expires.
- `expires_at` (optional): an absolute UTC timestamp instead of a day count, for clients
  that already compute one. Mutually exclusive with `expires_in_days`; sending both is 422.

Response is 201 with the key metadata plus the raw `token` (shown once), and
`Cache-Control: no-store` / `Pragma: no-cache` headers so nothing caches the body.

## List

`GET /users/api-keys[?include_revoked=true]`

Newest first, capped at 200 rows. Revoked keys are hidden unless `include_revoked=true` is
passed. Response header `X-API-Key-Limit` carries the active-key cap. Each item:

```json
{
  "id": "...",
  "name": "CI pipeline",
  "token_suffix": "a1b2c3",
  "status": "active",
  "created_at": "2026-09-18T10:00:00Z",
  "expires_at": "2026-12-17T10:00:00Z",
  "revoked_at": null,
  "last_used_at": "2026-09-18T11:30:00Z",
  "created_via": "oidc",
  "created_by_key_id": null,
  "created_by": null,
  "is_current": false
}
```

`is_current` is true for the row matching the key the request is authenticated with (always
false for a session-authenticated call). `token_suffix` and the raw token are never the
same thing: the suffix is the last 6 hex characters, kept only for display as `eve_…a1b2c3`.

## Revoke

`DELETE /users/api-keys/{id}`

204 on success, cascading to every key created by this one. 404 for a missing id, a
malformed id, or a key belonging to another user, all with the same body.

## Errors

| Status | Code | When |
| --- | --- | --- |
| 401 | (plain detail) | Missing, malformed or revoked bearer credential; a key used after its parent was revoked. |
| 403 | `PENDING_APPROVAL_DETAIL` shape | Caller's account is pending approval. |
| 404 | (plain detail) | Revoke on a missing, malformed or foreign key id. |
| 409 | `api_key_limit_reached` | Caller already has `API_KEY_MAX_ACTIVE_PER_USER` active keys. Body carries `limit`. |
| 422 | (Pydantic validation) | Bad name, bad `expires_in_days`, both expiry fields set, `expires_at` in the past or beyond the max lifetime. |
| 429 | `api_key_create_rate_limited` | More than `API_KEY_CREATE_MAX_PER_HOUR` creations in the last rolling hour. `Retry-After: 3600`. |

## Rotation

There is no in-place rotation endpoint. Rotate by creating a new key, updating the client
to use it, then revoking the old one. A key created for this purpose can use the same
`name` as the one it replaces; names are not unique.

## curl walkthrough

```bash
BASE=https://dev.eve-chat.chat/api

# 1. First key: from the web app (sidebar > API keys), or with a session access token.
curl -s -X POST "$BASE/users/api-keys" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "terminal key"}'
# -> 201, save the "token" field as $EVE_API_KEY; it is shown only this once.

# 2. Key-only create, no session needed from here on.
# No body at all: generated name, 90-day expiry.
curl -s -X POST "$BASE/users/api-keys" \
  -H "Authorization: Bearer $EVE_API_KEY"

# Never expires, clamped to the calling key's own expiry if it has one.
curl -s -X POST "$BASE/users/api-keys" \
  -H "Authorization: Bearer $EVE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"expires_in_days": null}'

# 3. List, including revoked.
curl -s "$BASE/users/api-keys?include_revoked=true" \
  -H "Authorization: Bearer $EVE_API_KEY"

# 4. Use the key.
curl -s "$BASE/users/me" -H "Authorization: Bearer $EVE_API_KEY"
curl -s "$BASE/v1/models" -H "Authorization: Bearer $EVE_API_KEY"
curl -s "$BASE/users/me/token-usage" -H "Authorization: Bearer $EVE_API_KEY"

curl -s -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $EVE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "eve", "messages": [{"role": "user", "content": "hi"}]}'

curl -sN -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $EVE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "eve", "messages": [{"role": "user", "content": "hi"}], "stream": true}'

# 5. Revoke. Children of this key stop working too (401), the key itself then answers 404.
curl -s -o /dev/null -w '%{http_code}\n' -X DELETE "$BASE/users/api-keys/<id>" \
  -H "Authorization: Bearer $EVE_API_KEY"
```

See [OpenAI-compatible gateway](openai-gateway.md) for `/v1/*` details: allowed paths,
budget charging, and the 429 shape when the shared token budget is exhausted.

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
