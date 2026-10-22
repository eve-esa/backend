## EVE Backend

A FastAPI-based backend service for chat with Retrieval-Augmented Generation (RAG). It provides authentication, collections and document ingestion, conversation/message management, streaming responses, and a hallucination detection pipeline.

For setup and running the backend, see:

- [Local development & configuration](https://eve-esa.github.io/eve-guide/backend/docs/local_setup/): prerequisites, environment variables, and running the backend directly on your machine
- [Docker setup](https://eve-esa.github.io/eve-guide/backend/docs/docker_setup/): run the backend with Docker Compose

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
  routers/            # FastAPI route handlers (user, collection, document, conversation, message, health_check, openai_proxy, mcp_proxy)
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
    - Sign in, sign up, verify mail and reset password at the identity provider (Keycloak
      locally, Cognito per AWS environment), not at this API
    - Machine callers use a self-service `eve_` API key instead: create, list, revoke via
      `routers.user`
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

- Auth (identity provider, API keys): [routers-auth](api/routers-auth.md)
- API keys (create/list/revoke, curl walkthrough): [api-keys](api/api-keys.md)
- OpenAI-compatible gateway (`/v1/*`, budget, SDK example): [openai-gateway](api/openai-gateway.md)
- Collections (public/private + examples): [routers-collection](api/routers-collection.md)
- Documents (ingestion + examples): [routers-document](api/routers-document.md)
- Conversations (chat lifecycle + examples): [routers-conversation](api/routers-conversation.md)
- Messages (generation/streaming + examples): [routers-message](api/routers-message.md)
- Generation failures (error codes, 429 and Retry-After): [generation-errors](api/generation-errors.md)
- Bug reports (`POST /bug-reports`, rate limit, log event): [bug-reports](api/bug-reports.md)

Use [swagger-api](api/swagger-api.md) only when you need exhaustive field-level reference.

### Base URLs

| Target | Base URL |
| --- | --- |
| Production API | `https://eve-chat.chat/api` |
| OpenAI-compatible gateway | `https://eve-chat.chat/api/v1` |
| Local backend | `http://localhost:8000` |

### Shared API setup

Use this once and reuse it across all API examples:

```python
import requests

BASE_URL = "https://eve-chat.chat/api"  # local: http://localhost:8000

# Bearer credential, either of:
# - an access token from the identity provider (sign in at https://eve-chat.chat)
# - an eve_ API key (see API keys)
TOKEN = "<access_token_or_eve_api_key>"
headers = {"Authorization": f"Bearer {TOKEN}"}
```

### Recommended API call order

1. **Auth first**
    - Sign in at the identity provider, get `ACCESS_TOKEN` from its response
    - Or create an `eve_` API key once with that token and use it from then on
2. **Collection discovery**
    - `GET /collections/public` to get valid `public_collections` names for generation
3. **Conversation lifecycle**
    - `POST /conversations` to get `conversation_id`
4. **Message generation**
    - `POST /conversations/{conversation_id}/generate-agentic` or `/stream-generate-agentic` ([agentic generation](api/routers-message.md#agentic-generation))
    - The classic `/messages` and `/stream_messages` routes are deprecated since 1.2 and will be removed in a later release
5. **Optional ingestion for private retrieval**
    - `POST /collections` -> get private `collection_id`
    - `POST /collections/{collection_id}/documents`
6. **Optional advanced/ops routes**
    - `.../retry`, `.../hallucination`, `/generate`, `/retrieve`, stats endpoints

### API dependency notes

- Message endpoints require a valid `conversation_id`.
- Generation endpoints require valid collection names (`public_collections`) from collection listing.
- Document endpoints require a valid private `collection_id`.
- Every route except `/health` and `/collections/public` requires `Authorization: Bearer <token>`, with an access token or an `eve_` API key.

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
