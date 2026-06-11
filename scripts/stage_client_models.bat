@echo off
REM Stage bundled HF models into data/models for client green package (dev helper).
setlocal
cd /d "%~dp0.."
if not exist "data\models\hub" mkdir "data\models\hub"
if exist ".hf_cache\hub\models--BAAI--bge-m3" (
  echo Copying bge-m3...
  xcopy /E /I /Y ".hf_cache\hub\models--BAAI--bge-m3" "data\models\hub\models--BAAI--bge-m3"
)
if exist ".hf_cache\hub\models--BAAI--bge-reranker-base" (
  echo Copying bge-reranker-base...
  xcopy /E /I /Y ".hf_cache\hub\models--BAAI--bge-reranker-base" "data\models\hub\models--BAAI--bge-reranker-base"
)
if exist ".hf_cache\hub\models--opendatalab--PDF-Extract-Kit-1.0" (
  echo Copying PDF-Extract-Kit MinerU...
  xcopy /E /I /Y ".hf_cache\hub\models--opendatalab--PDF-Extract-Kit-1.0" "data\models\hub\models--opendatalab--PDF-Extract-Kit-1.0"
)
echo Done. Models under data\models\hub
endlocal
