FROM python:3.11-slim

# Install system dependencies (CBC solver for PuLP, build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    coinor-libcbc-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY config/ ./config/
COPY src/ ./src/
COPY run_optimizer.py .
COPY run_optimizer .

# Create unprivileged user and ensure directory permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/players /app/data/templates /app/data/output /app/data/projections && \
    chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["python", "run_optimizer.py"]
CMD []
