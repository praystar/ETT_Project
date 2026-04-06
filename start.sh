#!/bin/bash
# Quick startup script for RAG Chatbot with Frontend

set -e

echo "🚀 Starting RAG Chatbot with Frontend..."
echo

# Check if .env exists (at project root)
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo "📝 Creating .env from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env - Please edit it with your API keys"
        echo
    else
        echo "❌ No .env.example found. Please create a .env file manually."
        exit 1
    fi
fi

# Check if Python venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "💡 Run: source .venv/bin/activate"
    echo
fi

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python -c "import flask" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}
echo "✅ Dependencies ready"
echo

# Start Flask backend in background
echo "🔧 Starting Flask backend..."
cd backend
python app.py &
FLASK_PID=$!
cd ..

# Wait for Flask to start
sleep 2

# Check if Flask started successfully
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo "❌ Failed to start Flask backend"
    exit 1
fi

echo "✅ Flask backend running (PID: $FLASK_PID)"
echo

# Open frontend in browser
echo "🌐 Opening frontend..."
FRONTEND_PATH="$(pwd)/frontend/index.html"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "file://$FRONTEND_PATH"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open &> /dev/null; then
        xdg-open "file://$FRONTEND_PATH"
    elif command -v firefox &> /dev/null; then
        firefox "file://$FRONTEND_PATH" &
    elif command -v chromium &> /dev/null; then
        chromium "file://$FRONTEND_PATH" &
    else
        echo "⚠️  Could not find browser. Open frontend manually:"
        echo "   file://$FRONTEND_PATH"
    fi
else
    # Windows or other
    echo "📂 Open this file in your browser:"
    echo "   $FRONTEND_PATH"
fi

echo
echo "==============================================="
echo "✅ RAG Chatbot is running!"
echo "==============================================="
echo
echo "📧 Backend API:  http://localhost:5000"
echo "🌐 Frontend:     file://$FRONTEND_PATH"
echo
echo "Press Ctrl+C to stop the server"
echo

# Wait for Flask process
wait $FLASK_PID
