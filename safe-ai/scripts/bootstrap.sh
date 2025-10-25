#!/bin/bash
# Bootstrap script for SAFE-AI Governor

set -e

echo "🚀 Bootstrapping SAFE-AI Governor..."

# Check dependencies
echo "Checking dependencies..."
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 required"; exit 1; }
command -v podman >/dev/null 2>&1 || command -v docker >/dev/null 2>&1 || { echo "❌ Podman or Docker required"; exit 1; }

# Create directories
echo "Creating directories..."
mkdir -p data/keys
mkdir -p logs
mkdir -p config

# Generate keys if not exist
if [ ! -f "data/keys/ed25519.default.sk" ]; then
    echo "Generating Ed25519 keypair..."
    python3 -c "
from pathlib import Path
from app.core.keys import KeyManager

km = KeyManager(Path('data/keys'))
keypair = km.generate_keypair('default')
km.save_keypair(keypair)
print('✅ Keys generated')
"
else
    echo "✅ Keys already exist"
fi

# Pull container image
RUNTIME="podman"
command -v podman >/dev/null 2>&1 || RUNTIME="docker"

echo "Pulling container image with $RUNTIME..."
$RUNTIME pull python:3.12-slim

# Configure rootless podman
if [ "$RUNTIME" = "podman" ]; then
    echo "Configuring rootless podman..."
    mkdir -p ~/.config/containers
    
    # Check subuid/subgid
    if ! grep -q "^$(whoami):" /etc/subuid 2>/dev/null; then
        echo "⚠️  Warning: User not in /etc/subuid. Run as root:"
        echo "    sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $(whoami)"
    fi
fi

# Create example config
if [ ! -f "config/settings.toml" ]; then
    echo "Creating example config..."
    cp app/config/settings.example.toml config/settings.toml
    echo "✅ Config created at config/settings.toml"
fi

# Verify installation
echo "Verifying installation..."
python3 -c "
from app.core.keys import KeyManager
from pathlib import Path
km = KeyManager(Path('data/keys'))
info = km.get_key_info('default')
print(f'✅ Verification successful - Key ID: {info.key_id}')
"

echo ""
echo "✅ Bootstrap complete!"
echo ""
echo "Next steps:"
echo "  1. Review config/settings.toml"
echo "  2. Create a manifest: examples/manifest.json"
echo "  3. Sign it: safeai sign -m manifest.json > signed.json"
echo "  4. Apply it: safeai apply -s signed.json -m manifest.json"
echo ""
echo "Or start the API server:"
echo "  uvicorn app.api.routes:app --host 0.0.0.0 --port 8000"
