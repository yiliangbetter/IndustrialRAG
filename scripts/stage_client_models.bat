@echo off
REM Stage every model declared by the portable-client runtime manifest.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."

set "MANIFEST=config\client_runtime_manifest.json"
if not exist "%MANIFEST%" (
  echo Missing client runtime manifest: %MANIFEST%
  exit /b 1
)
if not exist "data\models\hub" mkdir "data\models\hub"

set /a MODEL_COUNT=0
for /f "usebackq delims=" %%M in (`powershell -NoProfile -Command "(Get-Content -LiteralPath '%MANIFEST%' -Raw ^| ConvertFrom-Json).models"`) do (
  set /a MODEL_COUNT+=1
  call :stage_model "%%M"
)
if !MODEL_COUNT! equ 0 (
  echo No models found in %MANIFEST%
  exit /b 1
)

echo Done. Models under data\models\hub
exit /b 0

:stage_model
set "MODEL_NAME=%~1"
if exist ".hf_cache\hub\%MODEL_NAME%" (
  echo Copying %MODEL_NAME%...
  xcopy /E /I /Y ".hf_cache\hub\%MODEL_NAME%" "data\models\hub\%MODEL_NAME%" >nul
)
exit /b 0
