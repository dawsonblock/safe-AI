FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libsodium-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsodium23 \
    podman \
    fuse-overlayfs \
    slirp4netns && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash safeai && \
    mkdir -p /app /data /var/log/safeai && \
    chown -R safeai:safeai /app /data /var/log/safeai

# Configure podman for rootless
RUN echo "safeai:100000:65536" > /etc/subuid && \
    echo "safeai:100000:65536" > /etc/subgid

USER safeai
WORKDIR /app

# Copy application
COPY --chown=safeai:safeai --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --chown=safeai:safeai --from=builder /usr/local/bin /usr/local/bin
COPY --chown=safeai:safeai app ./app
COPY --chown=safeai:safeai pyproject.toml ./

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAFEAI_ENABLE=on \
    SAFEAI_DATA_DIR=/data \
    SAFEAI_LOG_DIR=/var/log/safeai

# Make filesystem read-only except for required writable paths
VOLUME ["/data", "/var/log/safeai", "/tmp"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/livez', timeout=5.0)"

EXPOSE 8000

# Run API server
CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
