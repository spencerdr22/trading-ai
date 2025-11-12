#!/bin/bash
set -e
docker-compose build
docker-compose up -d
docker exec -it trading-ai-app-1 python -m app.main --mode simulate --fast --symbol MES
