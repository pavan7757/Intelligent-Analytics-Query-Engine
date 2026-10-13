
@echo off
cd /d "%~dp0backend"
if not exist "venv\Scripts\python.exe" (
    echo Backend virtual environment missing. Run: python -m venv venv
    pause
    exit /b 1
)
venv\Scripts\python.exe -c "from google import genai; import fastapi, uvicorn, pandas, dotenv, httpx"
if errorlevel 1 (
    echo Backend dependencies missing. Run from backend:
    echo venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app
pause
