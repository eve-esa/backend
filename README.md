# EVE - Earth Virtual Expert

EVE (Earth Virtual Expert) is an AI-powered Digital Assistant designed to democratize access to Earth Observation and Earth Science knowledge. This open-source Large Language Model is specifically focused on Earth Observation, providing users with comprehensive access to EO data, insights, and expertise through conversational AI.

## About EVE

EVE is funded by ESA Φ-lab and developed by Pi School in collaboration with Imperative Space. The project aims to contribute to the development of a new generation of AI-powered 'Digital Assistant' interfaces for Earth Observation and Earth Science textual information.

### Project Objectives

- **Harness Earth Observation Knowledge** - Comprehensive access to EO data and insights
- **Launch Open-source LLM for EO** - Specialized language model focused on Earth Observation
- **Release Earth Observation Virtual Expert** - AI-powered assistant for EO professionals and enthusiasts
- **Equip Users with EO Knowledge** - Make Earth Science accessible to everyone
- **Compliant AI Development** - Responsible and ethical AI development practices

### Consortium

EVE is a collaborative effort between:
- **ESA Φ-lab** - European Space Agency's innovation lab (Funding)
- **Pi School** - AI education and research institute (Development)
- **Imperative Space** - Space technology company (Collaboration)

### Use Cases

EVE serves multiple user scenarios:

- **Beginner EO Concepts** - Understanding basic Earth Observation terms and concepts
- **Advanced EO Concepts** - Explanations of complex Earth Observation methodologies
- **EO Data Sources and Access** - Guidance on accessing and downloading Earth Observation data
- **Scientific Summarizations** - Automated summarization of Earth Observation scientific literature
- **Quick EO Insights** - Rapid access to Earth Science insights and critical information
- **General Conversational QA** - Wide variety of Earth Science and general questions

---

# EVE Backend Application

This repository contains the **backend application** for the EVE project.

- Getting Started
  - [Docker Setup](#docker-setup)
    - [Requirements](#requirements)
    - [Steps](#steps)
    - [Access API Documentation](#access-api-documentation)
    - [Access the backend container](#access-the-backend-container)
  - [Create User](#create-user)
  - [Run Tests](#run-tests)
  - [Format Code](#format-code)
  - [Code of Conduct](#code-of-conduct)
  - [Contributing](#contributing)
  - [License](#license)

## Getting Started

### Docker Setup

#### Requirements
- Docker
- Docker Compose

#### Steps

1. Clone the repository
2. Setup the environment variables `.env` [from the example](.env.example)
3. Adjust constant values in [constants.py](src/constants.py) if needed
4. Run `docker compose build`
5. Run `docker compose up -d`
6. You can now access the API at `http://localhost:8080`

#### Access API Documentation

Once the container is running, visit [http://localhost:8080/docs](http://localhost:8080/docs) to view the available endpoints.

#### Access the backend container

```bash
docker compose exec backend /bin/bash
```

### Create User

1. Inside the container, run the following command:

```bash
bash create_user.sh <email> <password>
```

### Run Tests

To run the tests, run the following command inside the container:

```bash
bash test.sh
```

### Format Code

To format the code, run the following command inside the container:

```bash
bash format.sh
```

### Code of Conduct

You can find the code of conduct [here](CODE_OF_CONDUCT.md).

### Contributing

You can find the contributing guidelines [here](CONTRIBUTING.md).

### License

You can find the license [here](LICENSE).
