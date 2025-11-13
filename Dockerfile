FROM python:3.12-slim

# Python 3.12 is the standard version for this project
# Minimum required: Python 3.12+

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