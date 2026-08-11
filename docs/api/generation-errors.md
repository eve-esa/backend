# Generation failure contract

How a chat generation failure reaches a client, on both channels. Introduced
2026-08-11 with the endpoint error handling fixes.

## The two channels

1. **Live (SSE)**: streaming endpoints emit a terminal event and close:

   ```json
   {"type": "error", "code": "timeout", "message": "..."}
   ```

2. **Durable (Mongo, read via `GET /conversations/{id}`)**: the message
   document carries `metadata.error` with the same projection:

   ```json
   {"code": "timeout", "type": "NodeTimeoutError", "message": "..."}
   ```

Both `message` fields are the same string (`str(exc)` truncated to 500 chars).
It is developer-facing diagnostics: user-visible copy derives from `code`
only, never from `message` or `type`.

## Codes

| code | Meaning | Suggested copy angle |
|---|---|---|
| `timeout` | The model did not produce a first token in time; almost always a serverless cold start | "warming up, retry in a moment" |
| `empty_answer` | Generation completed but produced nothing | "empty answer, retry" |
| `upstream_error` | Anything else from the model endpoint or pipeline | generic failure, retry |

The set is extensible and additive; codes are never renamed. Clients MUST
treat unknown codes as `upstream_error`.

## `metadata.endpoint`

A streaming turn walks an ordered chain of model endpoints
(`EVE_ENDPOINT_ORDER`) and skips the ones whose circuit a recent failure has
opened; non-streaming turns take the head of the chain only (no in-request
walk, and their failures do not feed the circuit). `metadata.endpoint` records
what happened, additively; no error code is involved when a turn that fell
through to the next endpoint succeeded. A mid-stream failure persists the
payload with `answered: null` and the failed attempt listed:

```json
{"requested": "eve_jsc",
 "chain": ["eve_jsc", "main", "fallback"],
 "answered": "main",
 "attempts": [{"llm_type": "eve_jsc", "outcome": "timeout"}],
 "circuit_open": ["eve_jsc"],
 "substituted": true}
```

- `requested` is the canonical llm_type the request asked for (legacy `runpod`
  and `mistral` read as `main` and `fallback`), or `null` when it asked for
  none.
- `chain` is the resolved candidate order: unconfigured endpoints are already
  dropped and open circuits already sunk to the back.
- `answered` is the endpoint that produced the output, and
  `generated_model_name` is its model.
- `attempts[].outcome` reuses the codes above.
- `circuit_open` lists candidates that were still cooling down when the turn
  started.
- `substituted` is true only when the request named an endpoint and a different
  one answered; a request that named none is never a substitution.

An explicit request is never promoted to another endpoint: asking for
`eve_jsc` can only be answered by `eve_jsc` or the fallback model.

Cancellation is NOT an error code: a user stop is carried by the SSE
`stopped` event and the `Message.stopped` field.

## Lifecycle

- The assistant message shell is created before generation with `output: ""`
  and, on agentic endpoints, `metadata.pipeline: "agentic"` (absent means
  classic or pre-contract).
- Every failure path persists `metadata.error` (replace, never merge).
- A successful retry (`POST .../messages/{id}/retry`) sets the new output,
  resets `stopped` to false and removes `metadata.error`. A failed retry
  overwrites `metadata.error` and returns HTTP 500 with
  `detail: {"code", "message"}`.
- A failed hallucination check writes `metadata.hallucination_error`
  (namespaced so it cannot mask or be cleared as a generation failure).

## Compatibility

- Documents written before this contract may carry `metadata.error` as a
  plain string: render it as a generic failure.
- Failed agentic messages from before the `pipeline` stamp have no trace and
  no stamp, and a retry routes them down the classic pipeline; accepted.
- Producers: `build_error_payload` / `build_empty_answer_payload` in
  `src/services/generate_answer.py` are the only sanctioned constructors.
