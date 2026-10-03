FROM python:3.12-slim-bookworm AS builder
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VIRTUAL_ENV
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Runtime layout shared by the production image and the local dev image.
FROM python:3.12-slim-bookworm AS runtime
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0
WORKDIR /code

COPY --from=builder /opt/venv /opt/venv

# Amazon DocumentDB TLS CA bundle: the cluster's CAs are AWS-private and not in any system
# trust store, so the client must be pointed at this file (tlsCAFile=/code/global-bundle.pem).
# https://docs.aws.amazon.com/documentdb/latest/devguide/connect_programmatically.html
ADD https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem /code/global-bundle.pem

COPY *.py ./
COPY config.yaml provider_models.yaml start.sh create_user.sh ./
RUN chmod +x start.sh create_user.sh
COPY src/ ./src/
COPY templates/ ./templates/

CMD ["./start.sh"]

# Local development only: the test suite and the docs toolchain on top of the runtime. Built only
# when asked for by name (compose build.target: dev); a build without a target builds the last stage.
# git: requirements-dev.txt includes requirements.txt, and pip re-checks its git dependency.
FROM runtime AS dev
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-dev.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/requirements-dev.txt && rm -rf /tmp/requirements

# Production. Must stay the last stage: the deploy workflow builds without a target.
FROM runtime AS prod

# Build-time identity. The image is built on push to main, before any version tag exists, and is
# then promoted to staging and prod by digest without being rebuilt, so the commit is the only
# thing that can be baked in here. The human version arrives at deploy time as APP_VERSION.
# Declared last on purpose: an ARG/ENV pair that changes on every commit must not invalidate the
# dependency or source layers above it.
ARG GIT_SHA
ENV APP_GIT_SHA=${GIT_SHA}
