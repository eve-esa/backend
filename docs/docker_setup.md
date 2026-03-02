## Docker setup with Docker Compose

This guide explains how to run the backend using Docker and Docker Compose.

Using Docker is useful when you want:

- A reproducible runtime environment
- To avoid installing Python and dependencies directly on your host

You still need access to external services like **MongoDB**, **Qdrant**, and **LLM providers**, either as containers or managed services.

### 1. Install Docker and Docker Compose

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

**macOS (Docker Desktop):**

1. Download Docker Desktop for Mac from the [official Docker website](https://www.docker.com/products/docker-desktop/).
2. Open the `.dmg` and drag Docker Desktop into `Applications`.
3. Launch Docker Desktop and complete the initial setup.
4. Verify installation:

```bash
docker --version
docker compose version
```

**Windows (Docker Desktop):**

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
- [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)

### 2. Clone the repository

```bash
git clone https://github.com/eve-esa/backend.git
cd backend
```

### 3. Configure environment variables for Docker

1. Copy the example file:

  ```bash
  cp .env.example .env
  ```

2. Edit `.env` for a Docker environment. Pay attention to:

    - **MongoDB**
        - When MongoDB is run as a Docker service in the same `docker-compose.yml`, set:

            ```env
            MONGO_HOST=mongo
            MONGO_PORT=27017
            ```

        - `MONGO_DATABASE` should match what you want for this stack (for example `eve-backend`).

    - **Qdrant / vector store**
        - `QDRANT_URL`
        - `QDRANT_API_KEY`

    - **LLM endpoints**
        - `MAIN_MODEL_URL`
        - `MAIN_MODEL_API_KEY`

    - **Embeddings**
        - `EMBEDDING_URL`
        - `EMBEDDING_API_KEY`

    - **Redis**
        - `REDIS_URL`: the same value you would use for local setup

Other environment variables (SMTP, Satcom, JWT options, etc.) are documented in the **Environment variable reference** section in `local_setup.md`.

### 4. Build and run the containers

From the project root:

```bash
docker compose build
docker compose up -d
```

This will:

- Build the backend image
- Start the backend (and any configured dependencies in `docker-compose.yml`)

Check that the containers are running:

```bash
docker compose ps
```

### 5. Access the API

By default the backend should be available at:

- `http://localhost:8000/docs`

(If you have customized ports in `docker-compose.yml` and .env, adjust the URL accordingly.)

### 6. Stopping and restarting

- Stop containers but keep them built:

```bash
docker compose down
```

- Restart after a change:

```bash
docker compose up -d --build
```

