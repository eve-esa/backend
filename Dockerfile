FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./*.py ./
COPY ./config.yaml ./
COPY ./src/ ./src/

EXPOSE 8080

CMD ["uvicorn", "server:app", "--workers", "2", "--host", "0.0.0.0", "--port", "8080"]
