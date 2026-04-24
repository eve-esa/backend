# Collection API

Collection routes are split into public collections (platform-managed) and private collections (user-owned).

## API call order

1. Call `GET /collections/public` to get valid public collection names.
2. Optionally call `POST /collections` to create a private collection.
3. Use these names/IDs in message generation and document upload routes.

Shared request setup is documented once in [API index](./index.md#shared-api-setup).

## List public collections

`GET /collections/public?page=1&limit=20`

::: routers.collection.list_public_collections
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/collections/public",
    params={"page": 1, "limit": 20},
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
public_collections = resp.json()["data"]
print([c["name"] for c in public_collections])
```

### Explanation

Returns selectable public collections. These names are required by generation endpoints.

### Notes

- Pass returned `name` values to `public_collections` in message/generate requests.

## List my private collections

`GET /collections?page=1&limit=20`

::: routers.collection.list_collections
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/collections",
    params={"page": 1, "limit": 20},
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
my_collections = resp.json()["data"]
print([(c["id"], c["name"]) for c in my_collections])
```

### Explanation

Lists private collections owned by the authenticated user.

### Notes

- Use returned IDs with document endpoints.

## Create private collection

`POST /collections`

::: routers.collection.create_collection
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.post(
    f"{BASE_URL}/collections",
    json={
        "name": "eo-missions-notes",
        "description": "Internal notes for Sentinel and Copernicus docs",
        "embeddings_model": "Qwen/Qwen3-Embedding-4B",
    },
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
collection = resp.json()
COLLECTION_ID = collection["id"]
print(collection)
```

### Explanation

Creates a user-owned collection for document ingestion and private retrieval.

### Notes

- `embeddings_model` is optional; default is server-defined.

## Get one collection

`GET /collections/{collection_id}`

::: routers.collection.get_collection
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/collections/{COLLECTION_ID}",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Returns collection metadata and counters such as document/vector totals.

### Notes

- Useful to confirm ingestion completed successfully.

## Rename collection

`PATCH /collections/{collection_id}`

::: routers.collection.update_collection
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.patch(
    f"{BASE_URL}/collections/{COLLECTION_ID}",
    json={"name": "eo-missions-notes-v2"},
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Updates mutable collection metadata.

### Notes

- Keep references in client code synchronized after rename.

## Delete collection

`DELETE /collections/{collection_id}`

::: routers.collection.delete_collection
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.delete(
    f"{BASE_URL}/collections/{COLLECTION_ID}",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Deletes the collection and removes related stored data.

### Notes

- Destructive operation; confirm ownership and usage before calling.

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
