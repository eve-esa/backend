# Getting Started with Backend

This guide will help you set up and run the backend on local machine or production.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.12+
- MongoDB
- Docker and Docker Compose

### Installing Prerequisites

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

**Windows:**

1. Download MongoDB Community Server from the [official MongoDB website](https://www.mongodb.com/try/download/community)
2. Run the installer and follow the setup wizard
3. MongoDB will be installed as a Windows service and start automatically

**Reference:** 
- [MongoDB Installation Guide - Ubuntu](https://www.mongodb.com/docs/manual/installation/)
- [MongoDB Installation Guide - Windows](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-windows/)

#### Docker and Docker Compose

**Ubuntu:**
```bash
# Remove old versions
sudo apt remove docker docker-engine docker.io containerd runc

# Install prerequisites
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine and Docker Compose
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version
```

**Windows:**

1. Download Docker Desktop from the [official Docker website](https://www.docker.com/products/docker-desktop/)
2. Run the installer and follow the setup wizard
3. Restart your computer if prompted
4. Docker Desktop includes both Docker and Docker Compose
5. Verify installation:

```cmd
docker --version
docker compose version
```

**Reference:**
- [Docker Installation Guide - Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)

## Local Development Setup
### 1. Create Virtual Environment
Create a virtual environment in the `venv` folder:
```
python3 -m venv venv
```
Activate the virtual environment:
```
source venv/bin/activate
```
### 2. Environment Configuration
Create a `.env` file in the root of the project with the following content:
```
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
INFERENCE_API_KEY=
SILICONFLOW_API_TOKEN=

SCRAPING_DOG_API_KEY=

SATCOM_SMALL_MODEL_NAME=esa-sceva/satcom-chat-8b
SATCOM_LARGE_MODEL_NAME=esa-sceva/satcom-chat-70b
SATCOM_LARGE_BASE_URL=https://api.runpod.ai/v2/zyy9iu4i7vmcxc/openai/v1
SATCOM_SMALL_BASE_URL=https://api.runpod.ai/v2/ucttr8up9sxh0k/openai/v1
SATCOM_RUNPOD_API_KEY=

SATCOM_QDRANT_URL=
SATCOM_QDRANT_API_KEY=

REDIS_URL=

IS_PROD=false
```

| Variable | Required | Description / Default |
| --- | --- | --- |
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
| `DEEPINFRA_API_TOKEN` | Yes | DeepInfra API token for reranking retrieved documents. |
| `INFERENCE_API_KEY` | Yes | Inference API key for embedding queries, used as a fallback. |
| `SILICONFLOW_API_TOKEN` | Yes | SiliconFlow API token for reranking, used as a fallback. |
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
| `REDIS_URL` | Yes | Redis connection string for pub/sub and cancellations. |
| `IS_PROD` | No | Set to `true` to enable production mode toggles. |

### 2.1. Obtaining API Keys and Tokens

#### Qdrant URL and API Key

**Qdrant Cloud (Recommended):**

1. Sign up for a free account at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a new cluster
3. Copy the cluster URL (e.g., `https://xxxxx-xxxxx-xxxxx.qdrant.io`)
4. Navigate to API Keys section and create a new API key
5. Use the cluster URL as `QDRANT_URL` and the API key as `QDRANT_API_KEY`

**Self-hosted Qdrant:**

- If running Qdrant locally, use `http://localhost:6333` as `QDRANT_URL`
- For self-hosted instances, API key may not be required (leave empty or check your Qdrant configuration)

**Reference:** [Qdrant Cloud Documentation](https://qdrant.tech/documentation/cloud/)

#### JWT Secret Key

Generate a secure random string for JWT token signing. You can use one of these methods:

**Using Python:**
```python
import secrets
print(secrets.token_urlsafe(32))
```

**Using OpenSSL (Linux/Mac):**
```bash
openssl rand -base64 32
```

**Using PowerShell (Windows):**
```powershell
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Maximum 256 }))
```

**Online Generator:**

- Use a secure random string generator like [randomkeygen.com](https://randomkeygen.com/)
- Copy a 256-bit key and use it as `JWT_SECRET_KEY`

**Important:** Keep this key secret and never commit it to version control.

#### Runpod API Key

1. Sign up for an account at [Runpod](https://www.runpod.io/)
2. Navigate to your account settings
3. Go to the API Keys section
4. Create a new API key
5. Copy the key and use it as `RUNPOD_API_KEY`

**Reference:** [Runpod API Documentation](https://docs.runpod.io/serverless/endpoints/quick-start)

#### Mistral API Key

1. Sign up for an account at [Mistral AI](https://mistral.ai/)
2. Navigate to your account dashboard
3. Go to API Keys section
4. Create a new API key
5. Copy the key and use it as `MISTRAL_API_KEY`

**Reference:** [Mistral AI API Documentation](https://docs.mistral.ai/)

#### Main and Fallback Model URLs

The system uses two LLM models: **MAIN** (primary) and **FALLBACK** (backup). Both must be configured with OpenAI-compatible API endpoints.

**For RunPod endpoints:**
```
MAIN_MODEL_URL=https://api.runpod.ai/v2/{endpoint_id}/openai/v1
```
Replace `{endpoint_id}` with your RunPod endpoint ID (e.g., `2f9o93xc90871m`).

**For localhost/self-hosted models:**
```
MAIN_MODEL_URL=http://localhost:8000/v1
```
Use the base URL of your OpenAI-compatible API endpoint.

**For Mistral (fallback):**
```
FALLBACK_MODEL_URL=https://api.mistral.ai/v1
```

**For other OpenAI-compatible services:**
Simply use the base URL of the service's OpenAI-compatible endpoint.

**Note:** Both URLs must be in OpenAI-compatible format. The model names can be optionally overridden using `MAIN_MODEL_NAME` and `FALLBACK_MODEL_NAME` environment variables, otherwise they default to values in `config.yaml`.

#### DeepInfra API Token

1. Sign up for an account at [DeepInfra](https://deepinfra.com/)
2. Navigate to your dashboard
3. Go to API Keys section
4. Create a new API token
5. Copy the token and use it as `DEEPINFRA_API_TOKEN`

**Reference:** [DeepInfra API Documentation](https://deepinfra.com/docs)

#### Inference API Key

The `INFERENCE_API_KEY` is used for embedding queries with the Qwen 3.4B embedding model via Inference.net.

1. Sign up for an account at [Inference.net](https://inference.net/register) (you can use GitHub, Google, or email)
2. After registration, you'll be redirected to your dashboard
3. Navigate to the API Keys section
4. Click on "Create API Key" to generate a new key
5. Copy the generated API key and use it as `INFERENCE_API_KEY`

**Reference:** [Inference.net Documentation](https://docs.inference.net/quickstart)

#### SiliconFlow API Token

1. Sign up for an account at [SiliconFlow](https://siliconflow.cn/)
2. Navigate to your account settings
3. Go to API Keys section
4. Create a new API token
5. Copy the token and use it as `SILICONFLOW_API_TOKEN`

**Reference:** [SiliconFlow Documentation](https://siliconflow.cn/docs)

#### Redis URL

**Redis Cloud (Recommended):**

1. Sign up for a free account at [Redis Cloud](https://redis.com/try-free/)
2. Create a new database
3. Copy the connection URL (format: `redis://:password@host:port`)
4. Use it as `REDIS_URL`

**Local Redis:**
- If running Redis locally: `redis://localhost:6379`
- If Redis has a password: `redis://:password@localhost:6379`

**Reference:** [Redis Cloud Documentation](https://redis.io/docs/cloud/)

#### ScrapingDog API Key (Optional)

1. Sign up for an account at [ScrapingDog](https://www.scrapingdog.com/)
2. Navigate to your dashboard
3. Copy your API key
4. Use it as `SCRAPING_DOG_API_KEY`

**Reference:** [ScrapingDog Documentation](https://docs.scrapingdog.com/)

### 3. Installation
```
pip install -r requirements.txt
```

### 4. Start the Server

```bash
chmod +x start.sh
./start.sh
```

The server will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Build and run the Containers
To run this backend using Docker, fix `MONGO_HOST=localhost` to `MONGO_HOST=mongo` in `.env`.
```bash
docker compose build
docker compose up -d
```