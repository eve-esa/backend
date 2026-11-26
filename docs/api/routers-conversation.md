# Conversation API

Conversation routes manage chat threads. Message generation routes require a conversation ID.

## API call order

1. Call `POST /conversations` to create a thread.
2. Use returned `conversation_id` with message endpoints.
3. Optionally list, fetch, rename, or delete conversations.

Shared request setup is documented once in [API index](https://eve-esa.github.io/eve-guide/backend/docs/).

## Create conversation

`POST /conversations`

::: routers.conversation.create_conversation
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/conversations",
    json={"name": "Copernicus onboarding"},
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
conversation = resp.json()
CONVERSATION_ID = conversation["id"]
print(conversation)
```

### Explanation

Creates a user-owned conversation thread.

### Notes

- Required before calling `POST /conversations/{conversation_id}/generate-agentic` or `/stream-generate-agentic` ([agentic generation](./routers-message.md#agentic-generation)). The classic `/messages` and `/stream_messages` routes are deprecated since 1.2 and will be removed in a later release.

## List my conversations

`GET /conversations?page=1&limit=20`

::: routers.conversation.list_conversations
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/conversations",
    params={"page": 1, "limit": 20},
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json()["data"])
```

### Explanation

Returns paginated conversations for the current user.

### Notes

- Use for chat history screens and conversation selection.

## Get one conversation

`GET /conversations/{conversation_id}`

::: routers.conversation.get_conversation
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
detail = resp.json()
print(detail["name"], len(detail["messages"]))
```

### Explanation

Returns conversation metadata with stored messages, and `active_message_id`: the id of the assistant message still being generated on this conversation, or `null` when no turn runs or the turn is past its deadline. When the store that tracks running turns does not answer, the value comes from the stored messages (see the notes).

### Notes

- Useful for restoring previous chat context.
- After a reload during a turn, the assistant message can be stored with no output and no error while the backend keeps generating. When `active_message_id` names that message, the turn is still running: poll this route until it turns `null`, then read the saved answer. `POST /conversations/{conversation_id}/stop` stops it.
- Poll while `active_message_id` names your last assistant message: the web app polls every 5 s. Once it is `null`, a last message still empty is read again on the primary database, so the answer holds the saved turn.
- The value is best effort. When the shared store that tracks running turns does not answer within half a second, the route does not wait or fail: it answers from the stored messages instead, and keeps doing so for the next 10 s without asking the store. In that case `active_message_id` is the last assistant message when it has no answer, no file, no error, was not stopped and is younger than the longest a turn can run, otherwise `null`. A turn that ended without saving can then show as running until it passes that age, and a turn that has just saved can show as running for one more poll. A turn that has just started can also read `null` for a few milliseconds on another worker.
- Without `REDIS_URL` each worker sees only the turns it runs itself, so with more than one worker the field is `null` on the others.

## Rename conversation

`PATCH /conversations/{conversation_id}`

::: routers.conversation.update_conversation_name
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.patch(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}",
    json={"name": "Copernicus onboarding v2"},
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Updates conversation display name.

### Notes

- Does not change message content.

## Delete conversation

`DELETE /conversations/{conversation_id}`

::: routers.conversation.delete_conversation
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.delete(
    f"{BASE_URL}/conversations/{CONVERSATION_ID}",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Removes the conversation and its related message records.

### Notes

- Destructive operation.

## Full API reference

For exhaustive schema details, use [Swagger API](https://eve-esa.github.io/eve-guide/backend-swagger/).
