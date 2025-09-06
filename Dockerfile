# Dockerfile with Multi-stage build (Final Corrected Version)

# ===================================================================
# STAGE 1: The "Builder" stage 🛠️
# ===================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install development dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application source code
COPY ./src /app/src
COPY setup.py .

# --- WE ARE REMOVING THE LINE BELOW ---
# The editable install is causing issues with the volume mount later.
# We will rely solely on PYTHONPATH.
# RUN pip install -e .


# ===================================================================
# STAGE 2: The "Final" stage 📦
# ===================================================================
FROM python:3.11-slim

WORKDIR /app

# Set standard Python environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Add the src directory to Python's search path
ENV PYTHONPATH=/app/src

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the application source code
COPY --from=builder /app/src /app/src

# --- WE ARE REMOVING THE LINES BELOW ---
# setup.py and requirements.txt are not needed in the final lean image
# if we are not running pip install.
# COPY --from=builder /app/requirements.txt /app/requirements.txt
# COPY --from=builder /app/setup.py /app/setup.py

# --- WE ARE REMOVING THE LINE BELOW ---
# RUN pip install -e .