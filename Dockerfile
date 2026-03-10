# Employee Attrition Prediction System - FastAPI Service
# Build: docker build -t attrition-api .
# Run:   docker run -p 8000:8000 attrition-api

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Ensure models directory exists (can be mounted at runtime)
RUN mkdir -p /app/models

# Expose API port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
