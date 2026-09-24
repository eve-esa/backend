#!/bin/bash
PORT=${PORT:-8000}
WORKERS=${WORKERS:-2}

# No --preload: each worker imports server.py after the fork, so the
# OpenTelemetry exporter threads started by init_telemetry are per worker.
exec gunicorn server:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:$PORT \
  --workers $WORKERS \
  --access-logfile - \
  --timeout 600
