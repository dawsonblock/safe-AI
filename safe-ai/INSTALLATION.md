# Installation & Verification Guide

This guide walks you through installing SAFE-AI Governor and verifying it works correctly.

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+) or macOS 12+
- **Python**: 3.12 or higher
- **Container Runtime**: Podman 3.0+ or Docker 20.10+
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Disk**: 1GB for installation, 10GB+ for data/logs
- **CPU**: 2 cores minimum, 4 cores recommended

### Software Dependencies
- `git`
- `python3-pip`
- `podman` or `docker`
- `curl` (for API testing)

## Installation Steps

### 1. Clone Repository

```bash
cd /path/to/projects
git clone https://github.com/safe-ai/governor.git safe-ai
cd safe-ai
```

### 2. Verify Python Version

```bash
python3 --version
# Should be 3.12.0 or higher
```

If Python 3.12 is not installed:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

**macOS:**
```bash
brew install python@3.12
```

### 3. Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate  # On macOS/Linux
# Or: venv\Scripts\activate  # On Windows
```

### 4. Install SAFE-AI Governor

```bash
# Install package
pip install -e .

# Verify installation
safeai --version
# Should output: safeai, version 1.0.0
```

### 5. Install Container Runtime

**Option A: Podman (Recommended)**

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y podman fuse-overlayfs slirp4netns
```

macOS:
```bash
brew install podman
podman machine init
podman machine start
```

**Option B: Docker**

Follow official docs: https://docs.docker.com/get-docker/

### 6. Configure Rootless Container Runtime

**For Podman:**

```bash
# Check subuid/subgid
grep "^$(whoami):" /etc/subuid /etc/subgid

# If not present, add (requires root):
sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $(whoami)

# Configure
mkdir -p ~/.config/containers
```

**For Docker:**

```bash
# Enable rootless mode
dockerd-rootless-setuptool.sh install
```

### 7. Run Bootstrap Script

```bash
./scripts/bootstrap.sh
```

This will:
- Create data directories
- Generate Ed25519 keypair
- Pull container image (python:3.12-slim)
- Create example config

### 8. Verify Installation

```bash
python3 scripts/validate.py
```

Expected output:
```
============================================================
SAFE-AI Governor Validation
============================================================

✓ Validating imports...
  ✓ All imports successful
✓ Validating structure...
  ✓ Structure valid
✓ Validating models...
  ✓ Models valid
✓ Validating cryptography...
  ✓ Cryptography valid
✓ Validating policy VM...
  ✓ Policy VM valid

============================================================
✓ All validations passed!
============================================================
```

## Post-Installation Configuration

### 1. Review Policy Configuration

Edit `app/config/policy.yaml` to customize:
- Allowed/denied paths
- Denied patterns
- Size limits

### 2. Set Environment Variables

Copy and customize:
```bash
cp .env.example .env
nano .env  # Edit as needed
```

Key variables:
- `SAFEAI_ENABLE=on` - Must be "on" to allow operations
- `SAFEAI_JWT_SECRET` - Change for production!
- `SAFEAI_POLICY_AUTONOMY_TIER` - Set to alpha/beta/gamma

### 3. Generate Production Keys

```bash
# Generate production keypair
safeai keys generate --key-id production

# Secure private key
chmod 600 data/keys/ed25519.production.sk
```

## Verification Tests

### Test 1: CLI Commands

```bash
# Test keys
safeai keys list

# Test policy (should fail - no manifest)
safeai plan -m examples/manifest.json || echo "Expected to fail"

# Test help
safeai --help
safeai keys --help
```

### Test 2: Create and Sign Manifest

```bash
# Create test directory
mkdir -p src

# Sign example manifest
safeai sign -m examples/manifest.json > signed.json

# Verify signature
safeai verify -i signed.json
```

Expected output:
```
✓ Signature valid
  Manifest hash: abc123...
  Targets: src/example.py
  Edits: 1
  Tests: 1
```

### Test 3: Policy Evaluation

```bash
safeai plan -m examples/manifest.json
```

Expected output:
```
✓ APPROVED
```

### Test 4: Apply Change

```bash
# Apply the example manifest
safeai apply -s signed.json -m examples/manifest.json
```

Expected output:
```
✓ Signature verified: abc123...
✓ Policy check passed
Running 1 test(s)...
✓ Tests passed
Applying changes...
✓ Applied: 1 files modified
✓ Audit record: #0
  Merkle root: def456...
```

### Test 5: Audit Log

```bash
# View audit log
safeai audit --tail 10

# Verify chain integrity
safeai audit --verify
```

Expected output:
```
✓ Audit chain valid
  Current root: def456...
```

### Test 6: Sandbox Execution

```bash
# Test sandbox
safeai sandbox run -- python3 --version
```

Expected output:
```
Running in sandbox: python3 --version
Python 3.12.x
```

### Test 7: API Server

```bash
# Start server (in background)
uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 &
sleep 3

# Test health endpoints
curl http://localhost:8000/livez
curl http://localhost:8000/readyz

# Test metrics
curl http://localhost:8000/metrics | head -20

# Stop server
pkill -f uvicorn
```

### Test 8: Docker Build

```bash
# Build image
docker build -t safe-ai-governor:test .

# Test image
docker run --rm safe-ai-governor:test python -c "import app; print('OK')"
```

## Common Issues & Solutions

### Issue: "No module named 'pydantic'"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall
pip install -e .
```

### Issue: "Container runtime not available"

**Solution:**
```bash
# Check runtime
podman --version  # or docker --version

# If not installed, install podman
sudo apt install podman
```

### Issue: "Permission denied" on keys

**Solution:**
```bash
# Fix permissions
chmod 600 data/keys/*.sk
chmod 644 data/keys/*.vk
```

### Issue: "Kill switch active"

**Solution:**
```bash
# Check status
ls -la KILL_SWITCH

# Remove if exists
rm KILL_SWITCH

# Check environment
echo $SAFEAI_ENABLE
# Should be "on"
```

### Issue: Sandbox timeout

**Solution:**
```bash
# Increase timeout
export SAFEAI_SANDBOX_TIMEOUT_SECONDS=600

# Or edit policy.yaml
```

### Issue: "Verify key not found"

**Solution:**
```bash
# Regenerate keys
safeai keys generate --key-id default

# Or run bootstrap
./scripts/bootstrap.sh
```

## Development Setup

For development with hot-reload:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run with coverage
pytest --cov=app --cov-report=html
open htmlcov/index.html

# Type checking
mypy app/

# Linting
ruff check app/

# Auto-format
ruff check --fix app/
```

## Production Checklist

Before deploying to production:

- [ ] Python 3.12+ installed
- [ ] Container runtime configured (rootless)
- [ ] Virtual environment created
- [ ] Package installed (`pip install -e .`)
- [ ] Keys generated with secure permissions
- [ ] Policy reviewed and customized
- [ ] Environment variables set (no defaults in production)
- [ ] JWT secret changed from default
- [ ] All validation tests pass
- [ ] Docker image builds successfully
- [ ] Audit log location configured with backup
- [ ] Monitoring configured (Prometheus)
- [ ] Logging configured (centralized)
- [ ] Network security (firewall, mTLS)
- [ ] Documentation reviewed
- [ ] Disaster recovery plan documented

## Uninstallation

To completely remove SAFE-AI Governor:

```bash
# Deactivate virtual environment
deactivate

# Remove installation
cd /path/to/safe-ai/..
rm -rf safe-ai/

# Remove data (CAUTION: includes audit log!)
rm -rf data/

# Remove container images
podman rmi python:3.12-slim safe-ai-governor:*
```

## Getting Help

If you encounter issues:

1. **Check logs**: `cat logs/safeai.log`
2. **Run validation**: `python3 scripts/validate.py`
3. **Review docs**: See README.md, SECURITY.md
4. **GitHub Issues**: https://github.com/safe-ai/governor/issues
5. **Security issues**: security@safe-ai.dev (private)

## Next Steps

After successful installation:

1. Read [QUICK_START.md](QUICK_START.md) for usage guide
2. Review [README.md](README.md) for complete documentation
3. Study [SECURITY.md](SECURITY.md) for security model
4. See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment

---

**Installation complete!** 🎉

You now have a fully functional SAFE-AI Governor system ready for gating AI-driven code changes.
