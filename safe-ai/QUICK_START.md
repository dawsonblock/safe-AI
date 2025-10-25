# Quick Start Guide

This guide gets you running SAFE-AI Governor in 5 minutes.

## Prerequisites

- Python 3.12+
- Podman or Docker (for sandbox)
- Linux (recommended) or macOS

## Installation

```bash
# Clone and enter directory
cd safe-ai/

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## First Run

### 1. Bootstrap

```bash
# Run bootstrap script
./scripts/bootstrap.sh

# This will:
# - Create data directories
# - Generate Ed25519 keypair
# - Pull container image
# - Create example config
```

### 2. Create Your First Manifest

Create `my_manifest.json`:

```json
{
  "targets": ["src/hello.py"],
  "edits": [
    {
      "path": "src/hello.py",
      "op": "replace",
      "start": 0,
      "end": 0,
      "text": "def hello():\n    print('Hello, SAFE-AI!')\n"
    }
  ],
  "tests": ["python src/hello.py"],
  "rationale": "Create hello world function for demonstration"
}
```

### 3. Sign and Apply

```bash
# Create src directory
mkdir -p src

# Sign manifest
safeai sign -m my_manifest.json > signed.json

# Verify signature
safeai verify -i signed.json

# Check policy
safeai plan -m my_manifest.json

# Apply change
safeai apply -s signed.json -m my_manifest.json

# View audit log
safeai audit --tail 10
```

### 4. Start API Server

```bash
# Development server
uvicorn app.api.routes:app --reload

# Production server
uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 --workers 4
```

API now available at http://localhost:8000

- Docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics
- Health: http://localhost:8000/livez

## Using the API

### Generate a Token (Development)

```python
from app.core.rbac import RBACManager, RBACConfig

config = RBACConfig(jwt_secret="development_secret_key")
rbac = RBACManager(config)
token = rbac.create_token("user123", "operator")
print(token)
```

### Make API Requests

```bash
# Set token
export TOKEN="your-jwt-token"

# Verify manifest
curl -X POST http://localhost:8000/v1/verify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @signed.json

# Apply change
curl -X POST http://localhost:8000/v1/apply \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "signed_envelope": '$(cat signed.json)',
    "manifest": '$(cat my_manifest.json)'
  }'

# Get audit records
curl http://localhost:8000/v1/audit/10 \
  -H "Authorization: Bearer $TOKEN"
```

## Docker Quick Start

```bash
# Build image
docker build -t safe-ai-governor .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e SAFEAI_ENABLE=on \
  -e SAFEAI_JWT_SECRET=your-secret \
  --name safeai \
  safe-ai-governor

# Check logs
docker logs safeai

# Generate keys inside container
docker exec safeai safeai keys generate
```

## Common Operations

### Key Management

```bash
# Generate new keypair
safeai keys generate --key-id production

# List keys
safeai keys list

# Rotate keys
safeai keys rotate --key-id default
```

### Sandbox Testing

```bash
# Run command in sandbox
safeai sandbox run -- pytest tests/

# With custom image
SAFEAI_SANDBOX_IMAGE=python:3.11-slim safeai sandbox run -- python --version
```

### Kill Switch

```bash
# Activate (emergency halt)
touch KILL_SWITCH

# Or via environment
export SAFEAI_ENABLE=off

# Deactivate
rm KILL_SWITCH
export SAFEAI_ENABLE=on
```

### Audit Operations

```bash
# View recent records
safeai audit --tail 50

# Verify chain integrity
safeai audit --verify

# Export to JSON
safeai audit --tail 1000 --json > audit_export.json
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run only unit tests
pytest app/tests/unit/

# Run only integration tests
pytest app/tests/integration/
```

## Troubleshooting

### "Verify key not found"

Generate keys first:
```bash
safeai keys generate
```

### "Container runtime not available"

Install podman or docker:
```bash
# Ubuntu/Debian
sudo apt install podman

# macOS
brew install podman
```

### "Permission denied" on keys

Fix permissions:
```bash
chmod 600 data/keys/*.sk
chmod 644 data/keys/*.vk
```

### "Kill switch active"

Deactivate:
```bash
rm KILL_SWITCH
export SAFEAI_ENABLE=on
```

## Next Steps

1. Review [README.md](README.md) for full documentation
2. Read [SECURITY.md](SECURITY.md) for threat model
3. Customize [app/config/policy.yaml](app/config/policy.yaml)
4. Set up monitoring with Prometheus
5. Configure RBAC with proper JWT secrets
6. Enable mTLS for production

## Support

- Issues: https://github.com/safe-ai/governor/issues
- Docs: https://safe-ai.dev/docs
- Security: security@safe-ai.dev
