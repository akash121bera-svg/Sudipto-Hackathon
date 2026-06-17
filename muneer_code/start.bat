@echo off
TITLE Autonomous Document Intelligence Agent
COLOR 0A
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║    AUTONOMOUS DOCUMENT INTELLIGENCE AGENT - STARTING        ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run: setup_demo.bat first
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/3] Starting FastAPI server on http://127.0.0.1:8000
echo.
echo  Dashboard:  http://127.0.0.1:8000
echo  API Docs:   http://127.0.0.1:8000/docs
echo.
echo [3/3] Press CTRL+C to stop the server.
echo.

python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

pause
