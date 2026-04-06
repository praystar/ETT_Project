@echo off
REM Quick startup script for RAG Chatbot with Frontend (Windows)

echo.
echo 🚀 Starting RAG Chatbot with Frontend...
echo.

REM Check if .env exists
if not exist ".env" (
    echo ⚠️  No .env file found!
    echo 📝 Creating .env from .env.example...
    if exist ".env.example" (
        copy .env.example .env
        echo ✅ Created .env - Please edit it with your API keys
        echo.
    ) else (
        echo ❌ No .env.example found. Please create a .env file manually.
        exit /b 1
    )
)

REM Check dependencies
echo 📦 Checking dependencies...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)
echo ✅ Dependencies ready
echo.

REM Start Flask backend
echo 🔧 Starting Flask backend...
start python app.py

REM Wait for Flask to start
timeout /t 2 /nobreak

echo ✅ Flask backend running
echo.

REM Open frontend in browser
echo 🌐 Opening frontend...
for /f "delims=" %%A in ('cd') do set "CURRENT_DIR=%%A"
set "FRONTEND_PATH=%CURRENT_DIR%\frontend.html"

start "" "%FRONTEND_PATH%"

echo.
echo ===============================================
echo ✅ RAG Chatbot is running!
echo ===============================================
echo.
echo 📧 Backend API:  http://localhost:5000
echo 🌐 Frontend:     file:///%FRONTEND_PATH:\=/%
echo.
echo Press Ctrl+C in the Flask terminal to stop the server
echo.

pause
