# Backend

A FastAPI based backend service for EVE. It is part of the Earth Virtual Expert (EVE) initiative — an open-science program funded by the European Space Agency’s Φ-lab and developed by Pi School, in collaboration with Imperative Space and Mistral AI.

## Earth Virtual Expert (EVE)

**Earth Virtual Expert (EVE)** aims to advance the use of Large Language Models (LLMs) within the Earth Observation (EO) and Earth Science (ES) community.

- Website: https://eve.philab.esa.int/  
- HuggingFace: https://huggingface.co/eve-esa
- Other repositories: https://github.com/eve-esa


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

## Funding

This project is supported by the European Space Agency (ESA) Φ-lab through the Large Language Model for Earth Observation and Earth Science project, as part of the Foresight Element within FutureEO Block 4 programme.

## Citation 

If you use this project in academic or research settings, please cite:

## License

This project is released under the Apache 2.0 License - see the [LICENSE](LICENSE) file for more details.

## Contributing

We welcome contributions!
Please open an issue or submit a pull request on GitHub to help improve the pipeline.