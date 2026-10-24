#!/bin/sh
# Start the FastAPI service (PORT is set by Render/Docker; defaults to 8000)
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
