FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY ../Project/I-Drill/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project structure
COPY ../Project/I-Drill .

# Set default environment variables
ENV PYTHONPATH=/app
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092