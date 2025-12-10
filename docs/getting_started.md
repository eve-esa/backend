# Getting Started with Backend

This guide will help you set up and run the backend on local machine or production.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.12+
- MongoDB
- Docker and Docker Compose

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

# API Keys
MISTRAL_API_KEY=
OPENAI_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
RUNPOD_API_KEY=

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
SATCOM_RUNPOD_API_KEY=

SATCOM_QDRANT_URL=
SATCOM_QDRANT_API_KEY=

REDIS_URL=

IS_PROD=false
```

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