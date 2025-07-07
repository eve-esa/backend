#!/bin/bash

uvicorn "server:app" --workers 2 --host 0.0.0.0 --port 8080
