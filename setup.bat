@echo off
rem One-time setup on Windows. Double-click this file.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
set RESULT=%ERRORLEVEL%
if not "%RESULT%"=="0" echo. & echo Setup did not finish. The message above says why.
pause
exit /b %RESULT%
