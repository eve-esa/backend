## Local development setup

This guide walks you through running the backend directly on your machine (without Docker).

This guide assumes you have the main prerequisites installed:

- Python 3.12+
- MongoDB
- (Optional) Docker & Docker Compose for containerized setup — see [Docker setup](docker_setup.md) for install instructions

If you need help installing Python or MongoDB, see the **Detailed installation commands for prerequisites** section at the bottom of this page.

### 1. Clone the repository

```bash
git clone https://github.com/eve-esa/backend.git
cd backend
```

### 2. Create and activate a virtual environment

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Configure environment variables

1. Copy the example file:

    ```bash
    cp .env.example .env
    ```

2. Edit `.env` and at minimum configure values for:

        - **Qdrant / vector store**
            - `QDRANT_URL`
            - `QDRANT_API_KEY` (can be empty for local/self‑hosted Qdrant without auth)
        - **LLM endpoints**
            - `MAIN_MODEL_URL`
            - `MAIN_MODEL_API_KEY`
        - **Embeddings**
            - `EMBEDDING_URL` (has a sensible default)
            - `EMBEDDING_API_KEY`
        - **MongoDB**
            - `MONGO_HOST` (usually `localhost`)
            - `MONGO_PORT` (usually `27017`)
            - `MONGO_DATABASE` (for example `eve-backend`)
        - **Auth**
            - `JWT_SECRET_KEY`

Other variables in the table in the **Environment variable reference** section below are **optional** for basic local development and can be configured later as you enable more features (SMTP, Satcom, external rerankers, etc.).

### 4. Install dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

### 5. Run MongoDB (local)

Make sure MongoDB is running:

- **macOS (Homebrew):** `brew services start mongodb-community@7.0`
- **Ubuntu:** `sudo systemctl start mongod`
- **Windows:** MongoDB service usually starts automatically after installation

### 6. Start the backend

```bash
chmod +x start.sh
./start.sh
```

The API and interactive docs should now be available at:

- `http://localhost:8000/docs`

---

### 7. Environment variable reference

Below is a more complete example of `.env` and a description of the most important variables. This applies to both local development and Docker-based setups.

```env
PORT=8000 # for docker
WORDER=2 # for docker
# QDRANT Configuration
QDRANT_URL=
QDRANT_API_KEY=

# LLM Model URLs (OpenAI-compatible format)
MAIN_MODEL_URL=https://api.runpod.ai/v2/2f9o93xc90871m/openai/v1
FALLBACK_MODEL_URL=https://api.mistral.ai/v1
# Optional: Override model names (defaults from config.yaml)
MAIN_MODEL_NAME=
FALLBACK_MODEL_NAME=

MAIN_MODEL_API_KEY=
FALLBACK_MODEL_API_KEY=

MODEL_TIMEOUT=13

EMBEDDING_URL=https://api.deepinfra.com/v1/openai
EMBEDDING_API_KEY=

EMBEDDING_FALLBACK_URL=https://api.siliconflow.com/v1
EMBEDDING_FALLBACK_API_KEY=

# MongoDB Configuration
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_USERNAME=root
MONGO_PASSWORD=
MONGO_DATABASE=eve-backend
MONGO_PARAMS=

# JWT Configuration
JWT_SECRET_KEY=
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# SMTP
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
EMAIL_FROM_ADDRESS=noreply@eve-esa.com
EMAIL_FROM_NAME=EVE

# CORS (comma separated list)
CORS_ALLOWED_ORIGINS=http://localhost:5173

DEEPINFRA_API_TOKEN=
SILICONFLOW_API_TOKEN=

SCRAPING_DOG_API_KEY=

SATCOM_SMALL_MODEL_NAME=esa-sceva/satcom-chat-8b
SATCOM_LARGE_MODEL_NAME=esa-sceva/satcom-chat-70b
SATCOM_LARGE_BASE_URL=https://api.runpod.ai/v2/zyy9iu4i7vmcxc/openai/v1
SATCOM_SMALL_BASE_URL=https://api.runpod.ai/v2/ucttr8up9sxh0k/openai/v1
SATCOM_RUNPOD_API_KEY=

SATCOM_QDRANT_URL=
SATCOM_QDRANT_API_KEY=

REDIS_URL=redis://127.0.0.1:6379/0

IS_PROD=false
```

| Variable | Required | Description / Default |
| --- | --- | --- |
| `PORT` | Yes (for docker) | Backend PORT (default: `8000`) |
| `WORKDER` | Yes (for docker) | Backend uvicon worker counts `2` |
| `QDRANT_URL` | Yes | Base URL for the primary Qdrant instance. |
| `QDRANT_API_KEY` | Yes | API key for the primary Qdrant instance. |
| `SATCOM_QDRANT_URL` | No | Base URL for the Satcom-specific Qdrant instance that is used when SatcomLLM is selected on staging. |
| `SATCOM_QDRANT_API_KEY` | No | API key for the Satcom-specific Qdrant instance that is used when SatcomLLM is selected on staging. |
| `MAIN_MODEL_URL` | Yes | OpenAI-compatible URL for the main LLM model (e.g., `https://api.runpod.ai/v2/{endpoint_id}/openai/v1` or `http://localhost:8000/v1`). |
| `FALLBACK_MODEL_URL` | Yes | OpenAI-compatible URL for the fallback LLM model (e.g., `https://api.mistral.ai/v1` or any OpenAI-compatible endpoint). |
| `MAIN_MODEL_NAME` | No | Model name for the main model (defaults to value in config.yaml). |
| `FALLBACK_MODEL_NAME` | No | Model name for the fallback model (defaults to value in config.yaml). |
| `MAIN_MODEL_API_KEY` | No | API key for the main model (falls back to `RUNPOD_API_KEY` if not set). |
| `FALLBACK_MODEL_API_KEY` | No | API key for the fallback model (falls back to `MISTRAL_API_KEY` if not set). |
| `MODEL_TIMEOUT` | Yes | Timeout for OpenAI setting |
| `EMBEDDING_URL` | Yes | Main Embedding Model(Qwen/Qwen3-Embedding-4B) provider url, OpenAI capatible (e.g., `https://api.deepinfra.com/v1/openai`) |
| `EMBEDDING_API_KEY` | Yes | Main Embedding Model provider API token |
| `EMBEDDING_FALLBACK_URL` | Yes | Fallback Embedding Model provider url, OpenAI capatible (e.g., https://api.siliconflow.com/v1) |
| `EMBEDDING_FALLBACK_API_KEY` | Yes | Fallback Embedding Model provider API token |
| `DEEPINFRA_API_TOKEN` | Yes | DeepInfra API token for embedding and reranking retrieved documents (recommended for best retrieval quality, but backend still works without it). |
| `SILICONFLOW_API_TOKEN` | Yes | SiliconFlow API token for reranking, used as a fallback only. |
| `SATCOM_RUNPOD_API_KEY` | No | Runpod key dedicated to Satcom workloads. |
| `MONGO_HOST` | Yes | MongoDB host (default `localhost` or `mongo` in docker). |
| `MONGO_PORT` | Yes | MongoDB port (default `27017`). |
| `MONGO_USERNAME` | No | MongoDB username (empty allowed for local). |
| `MONGO_PASSWORD` | No | MongoDB password. |
| `MONGO_DATABASE` | Yes | MongoDB database name (default `eve-backend`). |
| `MONGO_PARAMS` | No | Extra Mongo connection params (default `?authSource=admin`). |
| `JWT_SECRET_KEY` | Yes | Secret for signing JWTs. |
| `JWT_ALGORITHM` | No | JWT algorithm (default `HS256`). |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | No | Access token lifetime in minutes (default `15`). |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | No | Refresh token lifetime in days (default `7`). |
| `SMTP_HOST` | No | SMTP host (default `smtp.gmail.com`). |
| `SMTP_PORT` | No | SMTP port (default `587`). |
| `SMTP_USERNAME` | No | SMTP username. |
| `SMTP_PASSWORD` | No | SMTP password. |
| `EMAIL_FROM_ADDRESS` | No | Sender email address (default `noreply@eve-ai.com`). |
| `EMAIL_FROM_NAME` | No | Sender display name (default `EVE AI`). |
| `CORS_ALLOWED_ORIGINS` | No | Comma-separated list of allowed origins (default `http://localhost:5173`). |
| `SCRAPING_DOG_API_KEY` | No | API key for ScrapingDog service, used as fallback of retrieval. |
| `SATCOM_SMALL_MODEL_NAME` | No | Model name for Satcom small LLM. |
| `SATCOM_LARGE_MODEL_NAME` | No | Model name for Satcom large LLM. |
| `SATCOM_RUNPOD_API_KEY` | No | API key for Satcom Runpod workloads. |
| `EVE_JSC_BASE_URL` | No | OpenAI-compatible URL for the EVE-JSC model (required when `llm_type` is `eve_jsc`). |
| `EVE_JSC_MODEL_NAME` | No | Model name for EVE-JSC (default `alias-eve`). |
| `EVE_JSC_API_KEY` | No | API key for EVE-JSC, issued by Jülich (required when `llm_type` is `eve_jsc` or a model is called with the `jsc/` prefix; a blank value counts as unset). |
| `OPENAI_PROXY_UPSTREAM_URL` | No | OpenAI-compatible URL the `/v1/*` proxy forwards `eve`/`runpod` models to. Setting this or `EVE_JSC_BASE_URL` enables the proxy. |
| `OPENAI_PROXY_API_KEY` | No | API key for that upstream (falls back to `MAIN_MODEL_API_KEY`, which is the same RunPod endpoint). |
| `REDIS_URL` | Yes | Redis connection string for pub/sub and cancellations (optional; if not set, in-process cancellation is used)(default `redis://127.0.0.1:6379/0`). |
| `IS_PROD` | No | Set to `true` to enable production mode toggles. |
| `LOG_LEVEL` | No | Root log level: `DEBUG`, `INFO`, `WARNING` or `ERROR` (default `INFO`). `DEBUG` gives the verbose output the server used to run with. `httpx`, `httpcore`, `urllib3`, `pymongo`, `botocore`, `boto3`, `openai` and `mcp` stay at `WARNING` at every level, since they log request URLs and query bodies. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | OTLP HTTP collector base URL, e.g. `http://otel-collector:4318` with the stack `clickstack` profile. Unset or empty keeps telemetry off: no SDK import, no exporter thread, no middleware. |
| `OTEL_EXPORTER_OTLP_HEADERS` | No | Secret. Headers for the collector, `authorization=<ingest key>`. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | No | `http/protobuf`, the only protocol shipped. |
| `OTEL_RESOURCE_ATTRIBUTES` | No | Extra resource attributes; `deployment.environment.name=<env>` names the environment, otherwise `APP_ENVIRONMENT` is used. `service.name` is always `eve-backend`. |
| `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG` | No | Sampler, e.g. `parentbased_traceidratio` and `1.0`. SDK default `parentbased_always_on`. |
| `OTEL_SDK_DISABLED` | No | `true` turns telemetry off even with an endpoint set. |
| `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT` | No | Longest span attribute value (default `4096`). |
| `EVE_OTEL_CAPTURE_CONTENT` | No | Prompts, answers and tool arguments on GenAI spans. `true` only on dev and local. |
| `EVE_OTEL_EXPORT_USER_EMAIL` | No | Export the user email as a span attribute (default `false`). |
| `EVE_LANGFUSE_SCORES_ENABLED` | No | `true` posts thumbs feedback to Langfuse as scores on the message trace (`thumbs-<message_id>`, `hallucination-<message_id>`). Default `false`. Needs the three variables below and a message with `trace_id`. |
| `LANGFUSE_HOST` | No | Langfuse base URL for `POST /api/public/scores`, e.g. `http://langfuse-web:3000` inside the stack compose. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | No | Secret. Langfuse project keys, sent as Basic auth. Only used for scores; traces reach Langfuse through the collector. |
| `API_KEY_MAX_ACTIVE_PER_USER` | No | Max active self-service API keys per user (default `10`). `0` disables key creation. |
| `API_KEY_DEFAULT_EXPIRES_IN_DAYS` | No | Expiry applied when a create request omits it (default `90`). |
| `API_KEY_CREATE_MAX_PER_HOUR` | No | Rolling-hour cap on key creations per user (default `30`); `0` disables the throttle. |

---

### 8. Detailed installation commands for prerequisites (optional)

If you prefer copy‑pasteable installation commands, the following sections provide example steps for Python and MongoDB on common platforms. For Docker and Docker Compose, see [Docker setup](docker_setup.md). Always cross‑check with the official documentation for the latest instructions.

#### Python 3.12+

**Ubuntu:**

```bash
# Update package list
sudo apt update

# Install Python 3.12 and pip
sudo apt install python3.12 python3.12-venv python3-pip

# Verify installation
python3.12 --version
```

**macOS (Homebrew):**

1. Install Homebrew if you don't have it yet (see [brew.sh](https://brew.sh)).
2. Install Python:

```bash
brew install python@3.12
```

3. Verify installation:

```bash
python3 --version
```

**Windows:**

1. Download Python 3.12+ from the [official Python website](https://www.python.org/downloads/)
2. Run the installer and check "Add Python to PATH"
3. Verify installation:

```cmd
python --version
```

**Reference:** [Python Installation Guide](https://www.python.org/downloads/)

#### MongoDB

**Ubuntu:**

```bash
# Import MongoDB public GPG key
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Update package list and install MongoDB
sudo apt update
sudo apt install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod
sudo systemctl enable mongod
```

**macOS (Homebrew):**

```bash
# Tap the official MongoDB Homebrew repo
brew tap mongodb/brew

# Install MongoDB Community Edition
brew install mongodb-community@7.0

# Start MongoDB as a background service
brew services start mongodb-community@7.0
```

**Windows:**

1. Download MongoDB Community Server from the [official MongoDB website](https://www.mongodb.com/try/download/community)
2. Run the installer and follow the setup wizard
3. MongoDB will be installed as a Windows service and start automatically

**Reference:** 
- [MongoDB Installation Guide - Ubuntu](https://www.mongodb.com/docs/manual/installation/)
- [MongoDB Installation Guide - macOS](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/)
- [MongoDB Installation Guide - Windows](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/)
