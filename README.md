# EVE APIs

A FastAPI-based backend service for document management and AI-powered operations.

## Docker Setup

### Configure Environment Variables

Create a `.env` file with the required variables. Refer to the `env.example` file for the complete list of environment variables.

### Run the Container

```bash
docker compose build
docker compose up -d
```

### Access API Documentation

Once the container is running, visit [http://localhost:8000/docs](http://localhost:8000/docs) to view the available endpoints.

## Local Development Setup

To run the server locally, ensure you have a MongoDB instance running and configure the `.env` file with the appropriate variables.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Server

```bash
./start.sh
```

The server will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Create a new user

Inside the container, run the following command:

```bash
python -m src.commands.create_user test@gmail.com test
```


