# OpenAI-compatible gateway (`/v1/*`)

`src/routers/openai_proxy.py`. Lets any EVE-authenticated caller use an OpenAI SDK against
`https://dev.eve-chat.chat/api/v1` instead of the app's own conversation endpoints. Traffic
here is metered and charged against the caller's monthly token budget, same budget as chat.

## Auth

`Authorization: Bearer` with either an `eve_` API key (see [API keys](api-keys.md)) or a
provider access token. That credential is never relayed upstream: the proxy replaces it
with its own configured upstream key before forwarding.

## Allowed paths

Only these three, and only `POST`:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`

`GET /v1/models` is answered locally (not proxied) with the providers that are actually
configured; it lists `eve/<model>` and, where `EVE_JSC_API_KEY`/`EVE_JSC_BASE_URL` are set,
`jsc/<model>` too. Any other path or method under `/v1/*` answers `404` before any upstream
call is made; the match is exact, so a path that is not in canonical form is refused too.

## Model routing

Model names may carry a LiteLLM-style `<provider>/<model-id>` prefix:

- `eve/...` or `runpod/...` or a bare model id -> the RunPod upstream (default).
- `jsc/...` -> the Jülich upstream.

Only the recognised provider segment is stripped; the rest of the model id is forwarded
unchanged, e.g. `eve/eve-esa/EVE-Instruct` -> `eve-esa/EVE-Instruct` upstream.

## Budget charging

Every call to the three billable paths counts against the caller's token budget
(`src.services.token_rate_limiter`), the same budget the chat UI uses:

1. Before any upstream call, a rough estimate of the request is reserved against the cap
   with one atomic increment. Concurrent requests cannot all read the same stale usage and
   all get admitted past the cap.
2. The reservation is settled once the response is in (in a `finally`): the real charge
   replaces the estimate. A stream that is aborted mid-way is charged for what it already
   produced; a request that never got a response is refunded in full.
3. Non-streaming: charge from the upstream `usage` block, or an estimate if the upstream
   omits it. Streaming: usage is parsed out of the SSE chunks; `stream_options` is forced to
   include usage. A response status of 400 or above is not charged, only tracked.

## 429: budget exceeded

Checked before the upstream call (a 429 cannot follow `http.response.start`), so it never
costs an upstream request:

```json
{
  "detail": "Token budget exceeded for group 'default'. Limit is 100000 tokens per period. Resets at 2026-10-01T00:00:00+00:00.",
  "error": {
    "message": "Token budget exceeded for group 'default'. Limit is 100000 tokens per period. Resets at 2026-10-01T00:00:00+00:00.",
    "type": "insufficient_quota",
    "code": "token_budget_exceeded",
    "param": null
  }
}
```

Headers: `Retry-After` (seconds to the period reset) and `x-should-retry: false`, so an
OpenAI SDK's own retry logic backs off instead of hammering the gateway.

## OpenAI SDK example

```python
from openai import OpenAI

client = OpenAI(
    api_key="eve_...",  # an eve_ API key, or a provider access token
    base_url="https://dev.eve-chat.chat/api/v1",
)

resp = client.chat.completions.create(
    model="eve",
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="eve",
    messages=[{"role": "user", "content": "hi"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# JSC provider
resp = client.chat.completions.create(
    model="jsc/alias-eve",
    messages=[{"role": "user", "content": "hi"}],
)
```

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
