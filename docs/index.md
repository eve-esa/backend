## EVE Backend

A FastAPI-based backend service for chat with Retrieval-Augmented Generation (RAG). It provides authentication, collections and document ingestion, conversation/message management, streaming responses, and a hallucination detection pipeline.

For setup and running the backend, see:

- [Local development & configuration](https://eve-esa.github.io/eve-guide/backend/docs/local_setup/) — prerequisites, environment variables, and running the backend directly on your machine
- [Docker setup](https://eve-esa.github.io/eve-guide/backend/docs/docker_setup/) — run the backend with Docker Compose

For additional deployment details, you can also refer to [README.md](https://github.com/eve-esa/backend/blob/main/README.md).

### Architecture

- **FastAPI**: HTTP API and dependency injection
- **MongoDB/DocumentDB**: Primary datastore for users, collections, documents, conversations, messages
- **Qdrant**: Vector store for embeddings and retrieval
- **LLM providers**: Pluggable via `src/core/llm_manager.py`
- **Docs**: MkDocs Material + mkdocstrings (Google-style docstrings)

### Directory structure

```text
src/
  routers/            # FastAPI route handlers (auth, collection, document, conversation, message, tool, user, health)
  services/           # Business logic (auth, generate_answer, email, hallucination, etc.)
  database/           # ODM-style models and pagination helpers
  core/               # Vector store and LLM managers
  middlewares/        # Authentication dependencies
  schemas/            # Pydantic request/response models
  templates/          # Prompt templates and pipeline configs
  utils/              # Helpers, parsers, embeddings, rerankers
tests/                # API and domain tests
docs/                 # Site content (this page, api references)
```

### Key workflows

- **Authentication**
    - Signup, email activation, login, refresh
    - Endpoints in `routers.auth` and `routers.forgot_password`
- **Collections & Documents**
    - Create Qdrant collections, upload documents, delete documents
    - Ingestion triggers parsing, chunking, embedding, and vector upsert
    - Endpoints in `routers.collection` and `routers.document`
- **Conversations & Messages**
    - Create conversations, post messages, stream responses (SSE)
    - Retry message generation, update feedback/annotations
    - Endpoints in `routers.conversation` and `routers.message`
- **Hallucination Detection**
    - Synchronous detection and streaming modes
    - Annotates message metadata with label, reason, timings

### API guides (usage-first)

- Auth (working examples): `[routers-auth]`
- Collections (public/private + examples): `[routers-collection]`
- Documents (ingestion + examples): `[routers-document]`
- Conversations (chat lifecycle + examples): `[routers-conversation]`
- Messages (generation/streaming + examples): `[routers-message]`

Use `[swagger-api]` only when you need exhaustive field-level reference.

### Shared API setup

Use this once and reuse it across all API examples:

```python
import requests

BASE_URL = "http://localhost:8000"

# Auth flow:
# 1) signup -> verify -> login
# 2) set ACCESS_TOKEN from login response
ACCESS_TOKEN = "<your_access_token>"
headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
```

### Recommended API call order

1. **Auth first**
    - `POST /signup`
    - `POST /verify`
    - `POST /login` -> get `access_token` / `refresh_token`
2. **Collection discovery**
    - `GET /collections/public` to get valid `public_collections` names for generation
3. **Conversation lifecycle**
    - `POST /conversations` to get `conversation_id`
4. **Message generation**
    - `POST /conversations/{conversation_id}/messages` or `/stream_messages`
5. **Optional ingestion for private retrieval**
    - `POST /collections` -> get private `collection_id`
    - `POST /collections/{collection_id}/documents`
6. **Optional advanced/ops routes**
    - `.../retry`, `.../hallucination`, `/generate`, `/retrieve`, stats endpoints

### API dependency notes

- Message endpoints require a valid `conversation_id`.
- Generation endpoints require valid collection names (`public_collections`) from collection listing.
- Document endpoints require a valid private `collection_id`.
- Most non-auth endpoints require `Authorization: Bearer <access_token>`.

### Message generation flow (high-level)

1. Validate conversation ownership and requested collections
2. Expand requested collections with allowed public and user-owned collections
3. Optionally extract year range from filters for MCP usage
4. If starting a new chat, create `Conversation` first; then create placeholder `Message` record
5. Run the answer generation pipeline:
    - Build context (RAG decision, retrieval, reranking)
    - Generate answer from LLM and record timings and prompt metadata
6. Update `Message` output (answer), documents, flags, and latencies
7. Optionally schedule rollup/trim in background
8. For streaming endpoints, publish tokens and lifecycle events via bus

### Documentation notes

- Docstrings are Google-style with `Args:`, `Returns:`, `Raises:`.
- Module reference pages in `docs/` use:

```markdown
::: routers.<module>
```
