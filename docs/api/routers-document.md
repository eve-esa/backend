# Document API

Document routes ingest and manage files inside user-owned collections.

## API call order

1. Create/list a private collection from collection APIs.
2. Use `collection_id` with document upload/list/get/delete routes.
3. Use collection names in message/generate APIs for retrieval.

Shared request setup is documented once in [API index](https://eve-esa.github.io/eve-guide/backend/docs/).

## Upload documents to a collection

`POST /collections/{collection_id}/documents`

::: routers.document.upload_documents
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
files = [
    ("files", ("sentinel_overview.pdf", open("sentinel_overview.pdf", "rb"), "application/pdf")),
    ("files", ("copernicus_brief.txt", open("copernicus_brief.txt", "rb"), "text/plain")),
]

data = [
    ("metadata_urls", "https://example.org/sentinel_overview"),
    ("metadata_urls", "https://example.org/copernicus_brief"),
    ("metadata_names", "Sentinel Overview"),
    ("metadata_names", "Copernicus Brief"),
    ("chunk_size", "1024"),
    ("chunk_overlap", "100"),
]

resp = requests.post(
    f"{BASE_URL}/collections/{COLLECTION_ID}/documents",
    headers=headers,
    files=files,
    data=data,
    timeout=120,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Uploads and ingests one or more files into a target collection.

### Notes

- Supports multipart form data with repeated fields.
- `embeddings_model` is optional.

## List documents in a collection

`GET /collections/{collection_id}/documents?page=1&limit=20`

::: routers.document.list_documents
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.get(
    f"{BASE_URL}/collections/{COLLECTION_ID}/documents",
    params={"page": 1, "limit": 20},
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
documents = resp.json()["data"]
print([(d["id"], d["name"]) for d in documents])
```

### Explanation

Lists ingested documents for a collection.

### Notes

- Use returned IDs for document detail and delete routes.

## Get one document

`GET /collections/{collection_id}/documents/{document_id}`

::: routers.document.get_document
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
DOCUMENT_ID = documents[0]["id"]

resp = requests.get(
    f"{BASE_URL}/collections/{COLLECTION_ID}/documents/{DOCUMENT_ID}",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Returns metadata for a specific document in the collection.

### Notes

- Requires both `collection_id` and `document_id`.

## Delete one document

`DELETE /collections/{collection_id}/documents/{document_id}`

::: routers.document.delete_document
    options:
      show_root_heading: false
      show_source: false

### Usage

```python
resp = requests.delete(
    f"{BASE_URL}/collections/{COLLECTION_ID}/documents/{DOCUMENT_ID}",
    headers=headers,
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

### Explanation

Deletes a document and associated vectors.

### Notes

- Destructive operation; retrieval quality may change.

## Full API reference

For exhaustive schema details, use [Swagger API](./swagger-api.md).
