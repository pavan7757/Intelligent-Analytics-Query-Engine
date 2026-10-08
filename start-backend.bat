
@echo off
cd /d "%~dp0backend"
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app
pause
//python -m uvicorn app.main:app --reload --port 8000