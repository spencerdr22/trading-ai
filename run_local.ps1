Write-Output "Building and starting containers..."
docker-compose build
docker-compose up -d
docker exec -it trading-ai-app-1 python -m app.main --mode simulate --fast --symbol MES
