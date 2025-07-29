# EVE APIs

A FastAPI-based backend service for document management and AI-powered operations.

## Docker Setup

- In order to run the project with docker, you need to have docker and docker compose installed.

### Configure Environment Variables

Create a `.env` file with the required variables. Refer to the `env.example` file for the complete list of environment variables.

### Build and run the Containers

```bash
docker compose build
docker compose up -d
```

### Access API Documentation

Once the container is running, visit [http://localhost:8000/docs](http://localhost:8000/docs) to view the available endpoints.


### Access the backend container

```bash
docker compose exec backend /bin/bash
```

### Create a new user

Inside the container, run the following command:

```bash
python -m src.commands.create_user test@gmail.com test
```

### Run Tests

To run the tests, run the following command inside the container:

```bash
bash test.sh
```

## Local Development Setup without Docker

To run the server locally, ensure you have a MongoDB instance running and configure the `.env` file with the appropriate variables.

### Create python virtual environment

```bash
python -m venv venv
source venv/bin/activate
```


### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Server

```bash
chmod +x start.sh
./start.sh
```

### Run Tests

```bash
chmod +x test.sh
./test.sh
```

The server will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Create a new user

Inside the container, run the following command:

```bash
python -m src.commands.create_user test@gmail.com test
```


