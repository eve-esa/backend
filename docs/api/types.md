## Common

### Pagination

- **page**: int (default 1)
- **limit**: int (default 10)

### PaginationMetadata

- **total_count**: int
- **current_page**: int
- **total_pages**: int
- **has_next**: bool

### PaginatedResponse[T]

- **data**: list[T]
- **meta**: PaginationMetadata

## Auth

### LoginRequest

- **email**: string (email)
- **password**: string

### LoginResponse

- **access_token**: string
- **refresh_token**: string

### RefreshRequest

- **refresh_token**: string

### RefreshResponse

- **access_token**: string

### SignupRequest

- **email**: string (email)
- **password**: string
- **first_name**: string | null
- **last_name**: string | null

### SignupResponse

- **id**: string
- **email**: string (email)
- **first_name**: string | null
- **last_name**: string | null

### ResendActivationRequest

- **email**: string

### VerifyRequest

- **email**: string
- **activation_code**: string

## User

### UpdateUserRequest

- **first_name**: string
- **last_name**: string

### User (response model)

- Inherits `MongoModel` → includes **id**: string, **timestamp**: datetime
- **email**: string (email)
- **password_hash**: string
- **first_name**: string | null
- **last_name**: string | null
- **is_active**: bool
- **activation_code**: string | null

## Collections

### CollectionRequest

- **embeddings_model**: string
- **name**: string
- **description**: string

### CollectionUpdate

- **name**: string

### Collection (response model)

- Inherits `MongoModel` → includes **id**: string, **timestamp**: datetime
- **user_id**: string | null
- **name**: string
- **description**: string | null
- **embeddings_model**: string

## Documents

### AddDocumentRequest

- **embeddings_model**: string
- **chunk_size**: int
- **chunk_overlap**: int
- **metadata_urls**: list[string] | null
- **metadata_names**: list[string] | null

### UpdateDocumentRequest

- **embeddings_model**: string
- **source_name**: string
- **new_metadata**: dict | null

### Document (response model)

- Inherits `MongoModel` → includes **id**: string, **timestamp**: datetime
- **user_id**: string
- **collection_id**: string
- **name**: string
- **filename**: string | null
- **file_type**: string | null
- **source_url**: string | null
- **chunk_count**: int | null
- **file_size**: int | null
- **vector_ids**: list[string] | null

## Conversations

### ConversationCreate

- **name**: string

### ConversationNameUpdate

- **name**: string

### ConversationDetail (response model)

- **id**: string
- **user_id**: string
- **name**: string
- **timestamp**: datetime
- **messages**: list[Message]

### Conversation (response model)

- Inherits `MongoModel` → includes **id**: string, **timestamp**: datetime
- **user_id**: string
- **name**: string
- **summary**: string | null

## Messages

### GenerationRequest

- **query**: string
- **year**: list[int] | null
- **filters**: dict | null
- **llm_type**: string | null (one of 'runpod', 'mistral', 'satcom_small', 'satcom_large')
- **embeddings_model**: string
- **k**: int
- **temperature**: float
- **score_threshold**: float
- **max_new_tokens**: int
- **public_collections**: list[string]
- Note: server populates private `collection_ids` and `private_collections_map`.

### CreateMessageResponse

- **id**: string
- **query**: string
- **answer**: string
- **documents**: list[DocumentReference]
- **use_rag**: bool
- **conversation_id**: string
- **loop_result**: LoopResult | null
- **metadata**: ResponseMetadata

### DocumentReference

- **id**: string | null
- **version**: int | null
- **score**: float | null
- **reranking_score**: float | null
- **collection_name**: string | null
- **payload**: dict
- **text**: string
- **metadata**: dict

### Latencies

- **guardrail_latency**: float | null
- **rag_decision_latency**: float | null
- **query_embedding_latency**: float | null
- **qdrant_retrieval_latency**: float | null
- **mcp_retrieval_latency**: float | null
- **reranking_latency**: float | null
- **base_generation_latency**: float | null
- **fallback_latency**: float | null
- **hallucination_latency**: HallucinationLatencies | null
- **total_latency**: float | null

### HallucinationLatencies

- **detection_latency**: float | null
- **span_reprompting_latency**: float | null
- **query_rewriting_latency**: float | null
- **regeneration_latency**: float | null
- **overall_latency**: float | null

### ResponseMetadata

- **latencies**: Latencies

### LoopResult

- **final_answer**: string | null
- **generation_response**: dict | null
- **hallucination_response**: dict | null
- **rewrite_response**: dict | null
- **reflected_response**: dict | null
- **ranked_output**: dict | null
- **docs**: string | null

### HallucinationDetectResponse

- **label**: int
- **reason**: string

### SourceLogsRequest

- **source_id**: string | null
- **source_url**: string | null
- **source_title**: string | null
- **source_collection_name**: string | null

## Tools

### ToolConfigRequest

- **url**: string | null
- **transport**: "streamable_http" | "stdio" | null
- **headers**: dict[str, str] | null
- **command**: string | null
- **args**: list[string] | null
- **env**: dict[str, str] | null

### ToolRequest

- **name**: string
- **provider**: string | null
- **description**: string | null
- **type**: string (default "mcp")
- **enabled**: bool
- **environment**: list[string] | null
- **config**: ToolConfigRequest

### ToolUpdate

- Partial fields mirroring `ToolRequest`; all optional.

### Tool (response model)

- Inherits `MongoModel` → includes **id**: string, **timestamp**: datetime
- **user_id**: string | null
- **name**: string
- **provider**: string | null
- **description**: string | null
- **type**: enum ("mcp")
- **enabled**: bool
- **environment**: list[string] | null
- **config**: ToolConfig
- **created_at**: datetime
- **updated_at**: datetime
- **deleted_at**: datetime | null

### ToolConfig (response model)

- **url**: string | null
- **transport**: enum ("streamable_http" | "stdio") | null
- **headers**: dict[str, str] | null
- **command**: string | null
- **args**: list[string] | null
- **env**: dict[str, str] | null

## Forgot Password

### ForgotPasswordRequest

- **email**: string (email)

### ForgotPasswordConfirmation

- **new_password**: string
- **confirm_password**: string
- **code**: string
