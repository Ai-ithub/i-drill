FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements ./requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project structure
COPY . .

# Set default environment variables
ENV PYTHONPATH=/app
ENV KAFKA_BOOTSTRAP_SERVERS=localhost:9092