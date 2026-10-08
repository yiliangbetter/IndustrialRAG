@echo off
REM Build Nanxing RAG green portable package (offline models included).
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack_client.ps1" %*
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" exit /b %ERR%
endlocal
