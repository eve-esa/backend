# Message API

This page covers chat generation, streaming, retry, feedback, and testing endpoints.

## API call order

1. Authenticate first (`/signup` + `/verify` + `/login`).
2. Create conversation with `POST /conversations`.
3. Get valid collection names via `GET /collections/public` (and/or private collection names you own).
4. Call message or generate endpoints with `conversation_id` and collection names.
5. Optionally run retry, feedback, hallucination, and stats endpoints.

Shared request setup is documented once in [API index](./index.md#shared-api-setup).

## List public collections before generation

`GET /collections/public?page=1&limit=20`

::: routers.collection.list_public_collections
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
public_resp = requests.get(
    f"{BASE_URL}/collections/public",
    params={"page": 1, "limit": 20},
    headers=headers,
    timeout=30,
)
public_resp.raise_for_status()
available_public_collections = [c["name"] for c in public_resp.json()["data"]]
print(available_public_collections)
```

### Explanation

Fetches valid `public_collections` values for generation payloads.

### Notes

- Use returned names exactly as provided.

## Create message (non-streaming)

`POST /conversations/{conversation_id}/messages`

::: routers.message.create_message
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
payload = {
    "query": "Summarize Sentinel-1 mission goals and practical applications.",
    "public_collections": ["qwen-512-filtered", "wikipedia-512"],
    "k": 5,
    "temperature": 0.1,
    "score_threshold": 0.6,
    "llm_type": "main",
    "filters": {"must": [], "should": None, "must_not": None, "min_should": None},
}

resp = requests.post(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}/messages",
    json=payload,
    headers=headers,
    timeout=120,
)
resp.raise_for_status()
message = resp.json()
MESSAGE_ID = message["id"]
print(message["answer"])
```

### Explanation

Runs retrieval + generation and stores the response in the conversation.

### Notes

- Requires a valid `conversation_id`.
- Collection names should come from collection endpoints.

### Important params

- `query`: User prompt sent to the model.
- `score_threshold`: Retrieval similarity threshold from `0.0` to `1.0`.
- `k`: Number of retrieved documents from `0` to `10`.
- `filters`: Optional Qdrant-compatible filter object.
- `public_collections`: Collection names from collection listing endpoints.
- `temperature`: Generation temperature from `0.0` to `1.0`.
- `llm_type`: Optional model selector (for example `main`, `fallback`, `satcom_small`, `satcom_large`, `ship`, `eve_v05`).

## Create message (SSE streaming)

`POST /conversations/{conversation_id}/stream_messages`

::: routers.message.create_message_stream
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
with requests.post(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}/stream_messages",
    json={
        "query": "How is TROPOMI used to support policy making?",
        "score_threshold": 0.6,
        "temperature": 0.0645,
        "k": 10,
        "filters": {
            "should": None,
            "min_should": None,
            "must": [],
            "must_not": None
        },
        "llm_type": "main",
        "public_collections": [
            "Wiley AI Gateway",
            "esa-data-qwen-1024",
            "Wikipedia EO",
            "wikipedia-512",
            "satcom-chunks-collection",
            "qwen-512-filtered"
        ]
    },
    headers={**headers, "Accept": "text/event-stream"},
    stream=True,
    timeout=120,
) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if line:
            print(line)
```

### Explanation

Streams generated output as server-sent events.

### Notes

- Suitable for token-by-token UI updates.
- Payload fields are the same as `POST /conversations/{conversation_id}/messages`.

## Retry generation for one message

`POST /conversations/{conversation_id}/messages/{message_id}/retry`

::: routers.message.retry
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}/messages/{MESSAGE_ID}/retry",
    headers=headers,
    timeout=120,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Re-runs generation using the stored request input of that message.

### Notes

- Useful when model/provider transient failures occur.

## Update message feedback

`PATCH /conversations/{conversation_id}/messages/{message_id}`

::: routers.message.update_message
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.patch(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}/messages/{MESSAGE_ID}",
    json={
        "feedback": "positive",
        "feedback_reason": "Sources are relevant and accurate",
        "was_copied": True,
    },
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Stores user feedback and message-level annotations.

### Notes

- Call after displaying a generated response.

## Stop active generation

`POST /conversations/{conversation_id}/stop`

::: routers.message.stop_conversation
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}/stop",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Requests cancellation of an active generation stream in the conversation.

### Notes

- Usually paired with streaming UIs.

## Add source log for a message

`POST /conversations/{conversation_id}/messages/{message_id}/source_logs`

::: routers.message.get_source_logs
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}/messages/{MESSAGE_ID}/source_logs",
    json={
        "source_id": "doc-001",
        "source_url": "https://example.org/eo-doc",
        "source_title": "EO Mission Documentation",
        "source_collection_name": "qwen-512-filtered",
    },
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Adds source metadata associated with a generated answer.

### Notes

- Use when you want explicit source tracking beyond default retrieval metadata.

## Detect hallucination (non-streaming)

`POST /conversations/{conversation_id}/messages/{message_id}/hallucination`

::: routers.message.hallucination_detect
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}/messages/{MESSAGE_ID}/hallucination",
    headers=headers,
    timeout=120,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Runs hallucination detection and returns labels, reason, and related outputs.

### Notes

- Requires an existing message ID.

## Detect hallucination (SSE streaming)

`POST /conversations/{conversation_id}/messages/{message_id}/stream-hallucination`

::: routers.message.stream_hallucination
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
with requests.post(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}/messages/{MESSAGE_ID}/stream-hallucination",
    headers={**headers, "Accept": "text/event-stream"},
    stream=True,
    timeout=120,
) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if line:
            print(line)
```

### Explanation

Streams hallucination detection lifecycle events.

### Notes

- Useful for progressive moderation/validation UX.

## LLM-only generation

`POST /generate-llm`

::: routers.message.generate_llm
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/generate-llm",
    json={"query": "What is Earth Observation?"},
    headers=headers,
    timeout=60,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Runs direct LLM generation (main model path) without retrieval or conversation persistence.

### Notes

- Useful for baseline/debug scenarios.

## One-off full generate

`POST /generate`

::: routers.message.generate
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/generate",
    json={
        "query": "How is TROPOMI used to support policy making?",
        "score_threshold": 0.6,
        "temperature": 0.0645,
        "k": 10,
        "filters": {
            "should": None,
            "min_should": None,
            "must": [],
            "must_not": None
        },
        "llm_type": "main",
        "public_collections": [
            "Wiley AI Gateway",
            "esa-data-qwen-1024",
            "Wikipedia EO",
            "wikipedia-512",
            "satcom-chunks-collection",
            "qwen-512-filtered"
        ]
    },
    headers=headers,
    timeout=120,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Runs full retrieval + generation pipeline without storing a conversation message.

### Important params

- `query`: User prompt sent to the model.
- `score_threshold`: Retrieval similarity threshold from `0.0` to `1.0`.
- `k`: Number of retrieved documents from `0` to `10`.
- `filters`: Optional Qdrant-compatible filter object.
- `public_collections`: Collection names from collection listing endpoints.
- `temperature`: Generation temperature from `0.0` to `1.0` (lower is more deterministic).
- `llm_type`: Optional model selector (for example `main`, `fallback`, `satcom_small`, `satcom_large`, `ship`, `eve_v05`).

## Retrieval-only

`POST /retrieve`

::: routers.message.retrieve
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/retrieve",
    json={
        "query": "How is TROPOMI used to support policy making?",
        "score_threshold": 0.6,
        "k": 10,
        "filters": {
            "should": None,
            "min_should": None,
            "must": [],
            "must_not": None
        },
        "public_collections": [
            "Wiley AI Gateway",
            "esa-data-qwen-1024",
            "Wikipedia EO",
            "wikipedia-512",
            "satcom-chunks-collection",
            "qwen-512-filtered"
        ]
    },
    headers=headers,
    timeout=120,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Runs only retrieval and returns matched documents/metadata.

### Important params

- `query`: User query string.
- `score_threshold`: Retrieval similarity threshold from `0.0` to `1.0`.
- `k`: Number of retrieved documents from `0` to `10`.
- `filters`: Optional Qdrant-compatible filter object.
- `public_collections`: Collection names from collection listing endpoints.

## User message stats

`GET /conversations/messages/me/stats`

::: routers.message.get_my_message_stats
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/conversations/messages/me/stats",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Returns message-level usage statistics for the authenticated user.

### Notes

- Requires authentication.

## Average latency stats

`GET /conversations/messages/average-latencies`

::: routers.message.get_average_latencies
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/conversations/messages/average-latencies",
    params={"start_date": "2026-01-01T00:00:00Z", "end_date": "2026-12-31T23:59:59Z"},
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Returns average pipeline latency metrics for the selected date range.

### Notes

- Endpoint can be used for performance monitoring dashboards.

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
