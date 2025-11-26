## EVE Backend

A FastAPI-based backend service for chat with Retrieval-Augmented Generation (RAG). It provides authentication, collections and document ingestion, conversation/message management, streaming responses, and a hallucination detection pipeline. Documentation is generated via MkDocs + mkdocstrings.

For setup, Docker usage, local development, and deployment details, refer to [README.md](../README.md).

### Architecture

- **FastAPI**: HTTP API and dependency injection
- **MongoDB/DocumentDB**: Primary datastore for users, collections, documents, conversations, messages
- **Qdrant**: Vector store for embeddings and retrieval
- **LLM providers**: Pluggable via `src/core/llm_manager.py`
- **Docs**: MkDocs Material + mkdocstrings (Sphinx-style docstrings)

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

### API surface (reference)

- Auth: `[routers-auth]` — `routers.auth`
- Collections: `[routers-collection]` — `routers.collection`
- Documents: `[routers-document]` — `routers.document`
- Conversations: `[routers-conversation]` — `routers.conversation`
- Messages: `[routers-message]` — `routers.message`
- Users: `[routers-user]` — `routers.user`
- Health: `[routers-health_check]` — `routers.health_check`

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

### MCP information

This backend integrates with Model Context Protocol (MCP) servers, including Wiley’s MCP, using LangChain’s `MultiServerMCPClient`. The service automatically handles token acquisition and caches tokens for subsequent MCP calls.

#### How Wiley MCP works

- **Transport**: HTTP (streamable) or WebSocket sessions are established to the MCP endpoint, then a `ClientSession` is initialized and tools are invoked via `call_tool(name, arguments)`.
- **Auth model**: OAuth2 Client Credentials. The backend exchanges a Basic credential for a short‑lived Bearer access token, which is then sent on MCP requests.
- **Implementation**: See `src/services/mcp_client_service.py`:
  - Token fetch and caching in `_call_tool_over_network`
  - Session setup and `call_tool(...)` over HTTP/WebSocket
  - Server configuration loading in `_load_server_configs`

#### Which endpoint it exposes

- **MCP endpoint**: `https://custom-agents-dev-mcp.scholargateway.ai/mcp`
- **Token endpoint**: `https://custom-agents-dev-mcp.scholargateway.ai/oauth2/token?grant_type=client_credentials`

#### How to authenticate to Wiley MCP

1. Prepare client credentials:
   - Client ID: `XXX`
   - Client Secret: `XXX`
   - Authorization (Basic): `Basic base64(client_id:client_secret)`
2. Set the backend environment variable so the service can exchange for a Bearer token:
   - `WILEY_AUTH_TOKEN='Basic XXXXXX'`
3. The backend will:
   - POST to the Token endpoint with header `Authorization: $WILEY_AUTH_TOKEN`
   - Receive `{ access_token, expires_in }` and cache the token until `expires_in - 60s`
   - Send MCP requests with `Authorization: Bearer <access_token>`
   - For WebSocket, the Authorization is forwarded as query parameters `auth_header=Authorization` and `auth_value=Bearer%20<access_token>`

#### Operational notes

- Server definitions come from `Config.get_mcp_servers()`; ensure your Wiley MCP server is enabled with `transport` set to `streamable_http` or `websocket`, and `url` set to the MCP endpoint above.
- If an `Authorization` header is preconfigured without a `Bearer` or `Basic` prefix, the service normalizes it to `Bearer` at runtime.
- For examples of listing and invoking tools, see `MultiServerMCPClientService.list_tools_from_server(...)`, `list_tools_from_all_servers(...)`, and `call_tool_on_server(...)`.

### Documentation notes

- Docstrings are Sphinx-style with `:param`, `:type`, `:return`, `:rtype`, `:raises`.
- Module reference pages in `docs/` use:

```markdown
::: routers.<module>
handler: python
```
