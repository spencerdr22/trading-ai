# trading-ai/Dockerfile
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default command: run CLI help
CMD ["python", "-m", "app.main", "--help"]
