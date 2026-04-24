# Conversation API

Conversation routes manage chat threads. Message generation routes require a conversation ID.

## API call order

1. Call `POST /conversations` to create a thread.
2. Use returned `conversation_id` with message endpoints.
3. Optionally list, fetch, rename, or delete conversations.

Shared request setup is documented once in [API index](./index.md#shared-api-setup).

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

- Required before calling `POST /conversations/{conversation_id}/messages`.

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

Returns conversation metadata with stored messages.

### Notes

- Useful for restoring previous chat context.

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

For exhaustive schema details, use [Swagger API](./swagger-api.md).
