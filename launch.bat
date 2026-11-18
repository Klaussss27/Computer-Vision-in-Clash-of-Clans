@echo off
cd /d "%~dp0"
if not exist "config.json" copy /y "config.example.json" "config.json" >nul
if exist ".venv-ml\Scripts\python.exe" (
    ".venv-ml\Scripts\python.exe" dashboard.py
) else (
    python dashboard.py
)
if errorlevel 1 pause
