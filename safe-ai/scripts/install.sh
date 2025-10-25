#!/bin/bash
# Installation script for SAFE-AI Governor

set -e

echo "🔧 Installing SAFE-AI Governor..."
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package
echo "Installing SAFE-AI Governor..."
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
if command -v safeai &> /dev/null; then
    echo "✅ Installation successful!"
    safeai --version
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./scripts/bootstrap.sh"
    echo "  2. Try: safeai --help"
else
    echo "❌ Installation failed"
    exit 1
fi
