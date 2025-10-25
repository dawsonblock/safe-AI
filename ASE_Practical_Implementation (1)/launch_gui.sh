#!/bin/bash

# FDQC-Cockpit GUI Launcher
# Starts the production web dashboard

echo "🚀 Launching FDQC-Cockpit GUI..."
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "Please create .env file with your API keys"
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    cd gui
    pip3 install -r requirements.txt
    cd ..
fi

# Start the GUI
echo "✓ Starting GUI server..."
echo "✓ Dashboard will be available at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd gui
python3 app.py
