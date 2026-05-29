FROM python:3.12-slim-bookworm AS builder
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VIRTUAL_ENV
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim-bookworm AS prod
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0
WORKDIR /code

COPY --from=builder /opt/venv /opt/venv

COPY *.py ./
COPY config.yaml start.sh create_user.sh ./
RUN chmod +x start.sh create_user.sh
COPY src/ ./src/
COPY templates/ ./templates/

CMD ["./start.sh"]
