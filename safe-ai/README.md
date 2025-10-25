# SAFE-AI Governor v1.0

**Cryptographically-gated AI change management with policy VM and sandboxed execution**

[![CI](https://github.com/safe-ai/governor/workflows/CI/badge.svg)](https://github.com/safe-ai/governor/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

SAFE-AI Governor is a production-ready system for gating self-modifying AI operations through:
- **Ed25519 cryptographic signatures** on all changes
- **Policy VM** with configurable allow/deny rules
- **Rootless OCI sandboxes** (podman/docker) for test execution
- **Append-only audit log** with Merkle chain integrity
- **Global kill switch** for emergency halt
- **RBAC** via JWT tokens
- **Prometheus metrics** and structured logging

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/safe-ai/governor.git
cd governor

# Install dependencies
pip install -e .

# Or use Docker
docker build -t safe-ai-governor .
```

### 2. Generate Keys

```bash
# Generate Ed25519 keypair
safeai keys generate --key-id default

# List keys
safeai keys list
```

### 3. Create a Change Manifest

Create `manifest.json`:

```json
{
  "targets": ["src/module.py"],
  "edits": [
    {
      "path": "src/module.py",
      "op": "replace",
      "start": 10,
      "end": 15,
      "text": "def new_function():\n    return True\n"
    }
  ],
  "tests": ["pytest -q tests/"],
  "rationale": "Add new_function for feature X"
}
```

### 4. Sign and Apply

```bash
# Sign manifest
safeai sign -m manifest.json > signed.json

# Verify signature
safeai verify -i signed.json

# Check policy
safeai plan -m manifest.json

# Apply change
safeai apply -s signed.json -m manifest.json

# View audit log
safeai audit --tail 10
```

### 5. Run API Server

```bash
# Start server
uvicorn app.api.routes:app --host 0.0.0.0 --port 8000

# Or with Docker
docker run -p 8000:8000 -v $(pwd)/data:/data safe-ai-governor
```

## API Endpoints

### Health & Metrics

- `GET /livez` - Liveness probe
- `GET /readyz` - Readiness probe
- `GET /metrics` - Prometheus metrics

### Operations

All operations require JWT authentication via `Authorization: Bearer <token>` header.

- `POST /v1/verify` - Verify signed manifest
  ```bash
  curl -X POST http://localhost:8000/v1/verify \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d @signed.json
  ```

- `POST /v1/policy/plan` - Evaluate manifest against policy
  ```bash
  curl -X POST http://localhost:8000/v1/policy/plan \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d @manifest.json
  ```

- `POST /v1/apply` - Apply signed change
  ```bash
  curl -X POST http://localhost:8000/v1/apply \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"signed_envelope": {...}, "manifest": {...}}'
  ```

- `GET /v1/audit/{n}` - Get last N audit records
  ```bash
  curl http://localhost:8000/v1/audit/10 \
    -H "Authorization: Bearer $TOKEN"
  ```

- `GET /v1/audit/verify` - Verify audit chain integrity
  ```bash
  curl http://localhost:8000/v1/audit/verify \
    -H "Authorization: Bearer $TOKEN"
  ```

## Policy Configuration

Default policy (configurable via environment variables):

```yaml
safe_mode: true
autonomy_tier: gamma

allow_paths:
  - src/
  - tests/
  - app/

deny_paths:
  - .git/
  - config/keys/
  - /etc/
  - /usr/
  - secrets/

deny_patterns:
  - '(?i)subprocess\.Popen\('
  - 'rm\s+-rf'
  - 'curl\s+http'
  - 'wget\s+'
  - '(?i)ssh\s+'
  - 'pip\s+install'
  - '(?i)exec\('
  - '(?i)eval\('

allowed_operations:
  - replace
  - insert
  - delete

max_edit_size_bytes: 65536    # 64KB
max_total_size_bytes: 524288  # 512KB
```

Environment variables:
- `SAFEAI_POLICY_SAFE_MODE=true`
- `SAFEAI_POLICY_AUTONOMY_TIER=gamma`
- `SAFEAI_POLICY_ALLOW_PATHS=src/,tests/`
- `SAFEAI_POLICY_MAX_EDIT_SIZE_BYTES=65536`

## Sandbox Execution

All tests run in isolated OCI containers:

```bash
# Run command in sandbox
safeai sandbox run -- pytest tests/

# Configure sandbox
export SAFEAI_SANDBOX_IMAGE=python:3.12-slim
export SAFEAI_SANDBOX_CPU_LIMIT=1.0
export SAFEAI_SANDBOX_MEMORY_LIMIT=512m
export SAFEAI_SANDBOX_TIMEOUT_SECONDS=300
```

Features:
- Rootless podman/docker
- No network access by default
- Read-only project mount
- CPU and memory limits
- Timeout protection

## Kill Switch

Emergency halt mechanism:

```bash
# Via environment
export SAFEAI_ENABLE=off

# Via sentinel file
touch KILL_SWITCH

# Check status
curl http://localhost:8000/readyz
```

When active, all apply/rollback operations immediately fail with 503.

## RBAC

Three roles supported:
- **admin**: Full access including key rotation
- **operator**: Can verify, plan, apply, audit
- **auditor**: Read-only audit access

```python
# Generate token (example)
from app.core.rbac import RBACManager, RBACConfig

config = RBACConfig(jwt_secret="your-secret")
rbac = RBACManager(config)
token = rbac.create_token("user123", "operator")
```

## Audit Log

Append-only log with Merkle chain:

```bash
# View recent records
safeai audit --tail 50

# Verify chain integrity
safeai audit --verify

# Export as JSON
safeai audit --tail 100 --json > audit.json
```

Each record includes:
- Timestamp
- Actor (api/cli)
- Manifest SHA-256
- Policy verdict
- Sandbox results
- Merkle root and previous root
- Applied status

## Key Management

```bash
# Generate keypair
safeai keys generate --key-id production

# List keys
safeai keys list

# Rotate keys
safeai keys rotate --key-id production
```

Keys stored in `data/keys/`:
- `ed25519.{id}.sk` - Signing key (private, mode 0600)
- `ed25519.{id}.vk` - Verify key (public, mode 0644)

## Monitoring

Prometheus metrics available at `/metrics`:

```
safeai_policy_decisions_total{verdict="approved|blocked"}
safeai_policy_blocks_total{reason="..."}
safeai_signature_verifications_total{result="success|failure"}
safeai_apply_operations_total{result="success|failure"}
safeai_sandbox_executions_total{exit_code="0"}
safeai_sandbox_duration_seconds
safeai_kill_switch_active
safeai_api_requests_total{endpoint,method,status}
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy app/

# Linting
ruff check app/

# Run tests with coverage
pytest --cov=app --cov-report=html
```

## Deployment

### Docker

```bash
# Build
docker build -t safe-ai-governor:1.0.0 .

# Run
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e SAFEAI_ENABLE=on \
  -e SAFEAI_JWT_SECRET=your-secret \
  --name safeai \
  safe-ai-governor:1.0.0
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safe-ai-governor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: safe-ai-governor
  template:
    metadata:
      labels:
        app: safe-ai-governor
    spec:
      containers:
      - name: governor
        image: safe-ai-governor:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: SAFEAI_ENABLE
          value: "on"
        - name: SAFEAI_JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: safeai-secrets
              key: jwt-secret
        volumeMounts:
        - name: data
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /livez
            port: 8000
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8000
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: safeai-data
```

## Hardening Checklist

- [ ] All routes behind RBAC
- [ ] Kill switch checked before operations
- [ ] Unsigned manifests rejected
- [ ] Sandbox enforced for all tests
- [ ] Audit log immutable and chained
- [ ] Keys stored with proper permissions (0600)
- [ ] JWT secrets not in environment files
- [ ] mTLS enabled for production
- [ ] Rate limiting configured
- [ ] Monitoring and alerting active
- [ ] Regular key rotation scheduled
- [ ] Disaster recovery plan documented

## Threat Model

See [SECURITY.md](SECURITY.md) for detailed threat model and security considerations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI / API                            │
├─────────────────────────────────────────────────────────────┤
│  Kill Switch Check → Signature Verify → Policy Eval         │
│       ↓                                                      │
│  Sandbox Tests → Apply Atomically → Audit Log               │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
1. **Gate**: Ed25519 signature verification
2. **Policy VM**: Rule-based allow/deny engine
3. **Sandbox**: Rootless OCI container executor
4. **Applicator**: Atomic file operations with rollback
5. **Audit**: Merkle-chained append-only log
6. **Kill Switch**: Emergency halt mechanism

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests
4. Ensure CI passes
5. Submit pull request

## Support

- Issues: https://github.com/safe-ai/governor/issues
- Docs: https://safe-ai.dev/docs
- Security: security@safe-ai.dev
