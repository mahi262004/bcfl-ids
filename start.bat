@echo off
rem Start the demo on Windows. Double-click this file, and press Ctrl-C in this window to stop.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup first: double-click setup.bat
  pause
  exit /b 1
)
.venv\Scripts\python scripts\start_demo.py %*
