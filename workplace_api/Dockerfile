# Production Dockerfile for Workplace Passive-Aggression Detection API
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifacts
COPY . /app/workplace_api/

ENV PYTHONPATH=/app
ENV PORT=8001

EXPOSE 8001

CMD ["sh", "-c", "uvicorn workplace_api.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
