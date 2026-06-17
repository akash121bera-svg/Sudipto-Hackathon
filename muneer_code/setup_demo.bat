@echo off
title Autonomous Document Intelligence Setup
echo =====================================================================
echo  AUTONOMOUS DOCUMENT INTELLIGENCE AGENT PLATFORM SETUP
echo =====================================================================
echo.

:: Check for python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your PATH.
    echo Please install Python 3.10+ and run this script again.
    echo.
    pause
    exit /b
)

echo [1/3] Creating local virtual environment (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)

echo.
echo [2/3] Activating virtual environment and installing requirements...
call venv\Scripts\activate
pip install --upgrade pip
pip install -r backend/requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install pip packages.
    pause
    exit /b
)

echo.
echo [3/3] Running OCR Preprocessing and Extraction Verification...
python verify_ocr.py
if %errorlevel% neq 0 (
    echo [WARNING] Verification script failed. Check log outputs above.
)

echo.
echo =====================================================================
echo  SETUP COMPLETED SUCCESSFULLY!
echo =====================================================================
echo.
echo  Virtual environment is ready. To run the FastAPI server:
echo    1. Activate venv: call venv\Scripts\activate
echo    2. Start server:  uvicorn backend.app.main:app --reload
echo.
echo  Then open http://localhost:8000 in your browser to view the UI dashboard.
echo.
pause
