#!/bin/bash

APP_PORT=${PORT:-5000}

/opt/venv/bin/gunicorn -k uvicorn.workers.UvicornWorker --workers 9 --worker-connections 1000 src.app:app --bind "0.0.0.0:${APP_PORT}" --timeout 10000
#/opt/venv/bin/gunicorn -k uvicorn.workers.UvicornWorker --workers 3 --worker-class gevent --worker-connections 1000 src.app:app --bind "0.0.0.0:${APP_PORT}" --timeout 10000


#/opt/venv/bin/gunicorn src.app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5050 --timeout 600