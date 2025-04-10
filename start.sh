#!/bin/bash
PORT=${PORT:-8000}  # Use 8000 if PORT is unset
uvicorn server:app --host 0.0.0.0 --port "$PORT"