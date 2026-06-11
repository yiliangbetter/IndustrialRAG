#Requires -Version 5.1
<#
.SYNOPSIS
  Assemble Nanxing RAG green portable package with offline bundled models.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/pack_client.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/pack_client.ps1 -Zip -Force
#>
[CmdletBinding()]
param(
    [string]$OutDir = "dist/NanxingRAG",
    [switch]$Force,
    [switch]$Zip,
    [switch]$SkipModels,
    [switch]$SkipVenv,
    [switch]$SkipExe
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutRoot = Join-Path $RepoRoot $OutDir
$VenvPython = Join-Path $RepoRoot ".venv/Scripts/python.exe"
$RequiredScripts = @(
    "client_launcher.py",
    "client_paths.py",
    "client_env_manager.py",
    "client_setup_service.py",
    "rag_web_server.py",
    "rag_pipeline_parse_graph_chat.py",
    "query_doc_steering.py",
    "query_progress_hooks.py"
)
$RequiredModels = @(
    "models--BAAI--bge-m3",
    "models--BAAI--bge-reranker-base",
    "models--opendatalab--PDF-Extract-Kit-1.0"
)

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-RequiredModels([string]$ModelsHub) {
    $missing = @()
    foreach ($name in $RequiredModels) {
        $path = Join-Path $ModelsHub $name
        if (-not (Test-Path $path)) {
            $missing += $name
        }
    }
    if ($missing.Count -gt 0) {
        throw "Missing bundled models under $ModelsHub : $($missing -join ', '). Run scripts/stage_client_models.bat first."
    }
}

function Invoke-RobocopyMirror([string]$Source, [string]$Destination, [string[]]$ExtraArgs = @()) {
    if (-not (Test-Path $Source)) {
        throw "Source not found: $Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $args = @($Source, $Destination, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS") + $ExtraArgs
    & robocopy @args | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed ($LASTEXITCODE): $Source -> $Destination"
    }
}

Write-Step "Nanxing RAG client pack"
Write-Host "Repo: $RepoRoot"
Write-Host "Output: $OutRoot"

if (-not (Test-Path $VenvPython)) {
    throw "Dev venv not found. Run: uv sync --extra web --extra local-embed"
}

$ModelsHub = Join-Path $RepoRoot "data/models/hub"
if (-not $SkipModels) {
    if (-not (Test-Path $ModelsHub)) {
        Write-Step "Staging models into data/models (from .hf_cache)"
        & (Join-Path $RepoRoot "scripts/stage_client_models.bat")
    }
    Test-RequiredModels $ModelsHub
}

Write-Step "Validating bundled models via client_setup_service"
$validatePy = @"
import os, sys
sys.path.insert(0, r'$RepoRoot/scripts')
os.environ['RAG_CLIENT_MODE'] = '1'
os.environ['RAG_CLIENT_APP_ROOT'] = r'$RepoRoot'
from client_setup_service import check_bundled_models
rows = check_bundled_models()
bad = [r for r in rows if not r['ok']]
if bad:
    for r in bad:
        print('MISSING:', r['label'], '->', r['path'])
    sys.exit(1)
for r in rows:
    print('OK:', r['label'])
"@
& $VenvPython -c $validatePy
if ($LASTEXITCODE -ne 0) {
    throw "Model validation failed."
}

if ((Test-Path $OutRoot) -and -not $Force) {
    throw "Output already exists: $OutRoot`nUse -Force to overwrite."
}
if ((Test-Path $OutRoot) -and $Force) {
    Write-Step "Removing existing output"
    Remove-Item -LiteralPath $OutRoot -Recurse -Force
}

Write-Step "Creating directory layout"
$null = New-Item -ItemType Directory -Force -Path @(
    (Join-Path $OutRoot "scripts"),
    (Join-Path $OutRoot "web"),
    (Join-Path $OutRoot "config"),
    (Join-Path $OutRoot "data/rag_storage"),
    (Join-Path $OutRoot "data/pipeline_parse"),
    (Join-Path $OutRoot "logs")
)

Write-Step "Copying runtime scripts"
foreach ($name in $RequiredScripts) {
    Copy-Item -LiteralPath (Join-Path $RepoRoot "scripts/$name") -Destination (Join-Path $OutRoot "scripts/$name")
}

Write-Step "Copying web UI"
Invoke-RobocopyMirror (Join-Path $RepoRoot "web") (Join-Path $OutRoot "web")

Write-Step "Copying config template"
$envExampleSrc = Join-Path $RepoRoot "config/env.example"
if (-not (Test-Path $envExampleSrc)) {
    $envExampleSrc = Join-Path $RepoRoot "env.example"
}
Copy-Item -LiteralPath $envExampleSrc -Destination (Join-Path $OutRoot "config/env.example")
$steeringProfiles = Join-Path $RepoRoot "config/query_steering_profiles.json"
if (Test-Path $steeringProfiles) {
    Copy-Item -LiteralPath $steeringProfiles -Destination (Join-Path $OutRoot "config/query_steering_profiles.json")
}

if (-not $SkipModels) {
    Write-Step "Copying bundled models (this may take a while)"
    Invoke-RobocopyMirror (Join-Path $RepoRoot "data/models") (Join-Path $OutRoot "data/models")
}

if (-not $SkipVenv) {
    Write-Step "Copying Python runtime (.venv -> runtime/)"
    Invoke-RobocopyMirror (Join-Path $RepoRoot ".venv") (Join-Path $OutRoot "runtime") @("/XD", "Scripts/__pycache__")
}

$RuntimePython = Join-Path $OutRoot "runtime/Scripts/python.exe"
if (-not $SkipVenv) {
    Write-Step "Installing raganything into runtime (non-editable)"
    Copy-Item -LiteralPath (Join-Path $RepoRoot "pyproject.toml") -Destination (Join-Path $OutRoot "pyproject.toml")
    Invoke-RobocopyMirror (Join-Path $RepoRoot "raganything") (Join-Path $OutRoot "raganything")
    & uv pip install --python $RuntimePython --no-editable --no-deps $OutRoot
    if ($LASTEXITCODE -ne 0) {
        throw "uv pip install failed."
    }
} else {
    Write-Host "SkipVenv: skipped runtime copy and raganything install."
    $RuntimePython = $VenvPython
}

Write-Step "Building offline tiktoken cache"
$TiktokenDir = Join-Path $OutRoot "config/tiktoken_cache"
New-Item -ItemType Directory -Force -Path $TiktokenDir | Out-Null
$tiktokenPy = @"
import os, tiktoken
os.environ['TIKTOKEN_CACHE_DIR'] = r'$TiktokenDir'
tiktoken.get_encoding('cl100k_base')
print('tiktoken cache ready:', r'$TiktokenDir')
"@
& $RuntimePython -c $tiktokenPy

Write-Step "Writing NanxingRAG.bat launcher"
$bat = @"
@echo off
setlocal
cd /d "%~dp0"
set RAG_CLIENT_MODE=1
set RAG_CLIENT_APP_ROOT=%~dp0
set HF_EMBED_OFFLINE=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TIKTOKEN_CACHE_DIR=%~dp0config\tiktoken_cache
"%~dp0runtime\Scripts\python.exe" "%~dp0scripts\client_launcher.py"
if errorlevel 1 pause
endlocal
"@
Set-Content -LiteralPath (Join-Path $OutRoot "NanxingRAG.bat") -Value $bat -Encoding ASCII

if (-not $SkipExe) {
    Write-Step "Building NanxingRAG.exe (thin launcher, PyInstaller)"
    & uv pip install pyinstaller --python $VenvPython | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install pyinstaller in dev venv."
    }
    $PyWork = Join-Path $RepoRoot "build/pyinstaller"
    $PyDist = Join-Path $RepoRoot "build/pyinstaller-dist"
    New-Item -ItemType Directory -Force -Path $PyWork, $PyDist | Out-Null
    $LauncherExeSrc = Join-Path $RepoRoot "scripts/client_launcher_exe.py"
    & $VenvPython -m PyInstaller `
        --noconfirm --clean `
        --onefile --console `
        --name NanxingRAG `
        --distpath $PyDist `
        --workpath $PyWork `
        --specpath $PyWork `
        $LauncherExeSrc
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
    $BuiltExe = Join-Path $PyDist "NanxingRAG.exe"
    if (-not (Test-Path $BuiltExe)) {
        throw "Expected exe not found: $BuiltExe"
    }
    Copy-Item -LiteralPath $BuiltExe -Destination (Join-Path $OutRoot "NanxingRAG.exe") -Force
    Write-Host "NanxingRAG.exe -> $(Join-Path $OutRoot 'NanxingRAG.exe')"
} else {
    Write-Host "SkipExe: NanxingRAG.exe not built (use NanxingRAG.bat)."
}

Copy-Item -LiteralPath (Join-Path $RepoRoot "docs/client/README.txt") -Destination (Join-Path $OutRoot "README.txt")

Write-Step "Package summary"
$sizeGb = [math]::Round(((Get-ChildItem $OutRoot -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB), 2)
Write-Host "Output folder: $OutRoot"
Write-Host "Approx size: ${sizeGb} GB"
Write-Host "Launch: NanxingRAG.exe (primary) or NanxingRAG.bat (fallback)"
Write-Host "Customer: no Python / uv / pip / command line required"
Write-Host "Models: offline under data/models/hub (no Hugging Face required at customer site)"

if ($Zip) {
    Write-Step "Creating zip archive (slow for large packages)"
    $zipPath = "$OutRoot.zip"
    if ((Test-Path $zipPath) -and $Force) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    if (Test-Path $zipPath) {
        throw "Zip already exists: $zipPath (use -Force)"
    }
    Compress-Archive -LiteralPath $OutRoot -DestinationPath $zipPath -CompressionLevel Optimal
    $zipGb = [math]::Round((Get-Item $zipPath).Length / 1GB, 2)
    Write-Host "Zip: $zipPath (${zipGb} GB)"
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
