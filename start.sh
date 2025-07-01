#!/bin/bash
PORT=${PORT:-8000}
WORKERS=${WORKERS:-2}

exec gunicorn server:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:$PORT \
  --workers $WORKERS \
  --timeout 60