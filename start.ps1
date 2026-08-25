# Launch the Study Coach app.
#
#     .\start.ps1              chat app (study-coach)
#     .\start.ps1 -Rag         ingestion UI (textbook-rag) instead
#     .\start.ps1 -Check       verify setup and exit, don't launch
#
# Replaces setting $py and five environment variables by hand every time a
# terminal is opened. Secrets come from .env, which is gitignored.
#
# NOTE: this file is deliberately pure ASCII. Windows PowerShell 5.1 reads .ps1
# as ANSI unless the file has a UTF-8 BOM, so a stray em-dash or curly quote
# becomes mojibake and breaks the parser.

param(
    [switch]$Rag,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# --- virtual environment --------------------------------------------------
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No virtual environment found at .venv" -ForegroundColor Yellow
    Write-Host "Create it once with:" -ForegroundColor Yellow
    Write-Host "    python -m venv .venv"
    Write-Host "    .\.venv\Scripts\Activate.ps1"
    Write-Host "    pip install -r requirements.txt"
    exit 1
}

# --- secrets from .env ----------------------------------------------------
# Plain KEY=value lines. Quotes are stripped so both styles work, and anything
# already set in the shell wins so you can override for a single run.
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $idx = $line.IndexOf("=")
            $key = $line.Substring(0, $idx).Trim()
            $val = $line.Substring($idx + 1).Trim().Trim('"').Trim("'")
            if (-not [Environment]::GetEnvironmentVariable($key)) {
                [Environment]::SetEnvironmentVariable($key, $val)
            }
        }
    }
    Write-Host "loaded .env" -ForegroundColor DarkGray
}
else {
    Write-Host "no .env found - copy .env.example to .env and fill it in" -ForegroundColor Yellow
}

# --- report what's configured (never print the secrets themselves) --------
if ($env:LLM_PROVIDER) { $provider = $env:LLM_PROVIDER } else { $provider = "huggingface (default)" }
if ($env:STORE_BACKEND) { $store = $env:STORE_BACKEND } else { $store = "file (default)" }
$hasKey = ($env:GROQ_API_KEY -or $env:HF_TOKEN -or $env:LLM_API_KEY -or $env:GEMINI_API_KEY)
if ($hasKey) { $keyState = "set" } else { $keyState = "MISSING" }
if ($env:PG_DSN) { $dbState = "set" } else { $dbState = "not set" }

Write-Host ""
Write-Host "provider : $provider"
Write-Host "store    : $store"
Write-Host "api key  : $keyState"
Write-Host "database : $dbState"
if ($env:PAGE_LIMIT) {
    Write-Host "pages    : limited to $($env:PAGE_LIMIT) (trial run)" -ForegroundColor Yellow
}
Write-Host ""

if (-not $hasKey) {
    Write-Host "No API key set. The app will start but cannot talk to a model." -ForegroundColor Yellow
}

if ($Check) {
    & $python -c "import streamlit, langgraph, docling, sentence_transformers, gtts; print('all packages import OK')"
    exit $LASTEXITCODE
}

# --- launch ---------------------------------------------------------------
if ($Rag) {
    $target = "textbook-rag\app.py"
    $port = "8502"
}
else {
    $target = "study-coach\app.py"
    $port = "8501"
}

Write-Host "starting $target on port $port ..." -ForegroundColor Green
& $python -m streamlit run (Join-Path $root $target) --server.port $port
