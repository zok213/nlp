# NLP Exam Solver - One-Click Installer
# ======================================
#
# ONE COMMAND: .\install.ps1
#
# This script does EVERYTHING:
#   1. Install Ollama (if not present)
#   2. Download GGUF model from Google Drive (~4.4 GB)
#   3. Create Ollama model (exam-r1-solver)
#   4. Pull vision model (qwen2.5vl:3b)
#   5. Configure OpenCode integration
#   6. Install Python dependencies
#   7. Test everything
#
# Run in PowerShell as Administrator

param(
    [switch]$SkipOllama,
    [switch]$SkipVision,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

# Colors
function Write-Step { param($msg) Write-Host "`n[STEP] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "[FAIL] $msg" -ForegroundColor Red }
function Write-Progress { param($msg, $pct) Write-Host "       $msg ($pct%)" -ForegroundColor Gray }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir
$ModelsDir = Join-Path $ProjectRoot "models"
$DataDir = Join-Path $ProjectRoot "data"

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  NLP Exam Solver - Full Installer" -ForegroundColor Magenta
Write-Host "  One command, everything installed" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

# ============================================================
# STEP 1: Setup directories
# ============================================================
Write-Step "Setting up directories..."

foreach ($dir in @($ModelsDir, $DataDir, (Join-Path $ProjectRoot "outputs"))) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Success "Directories ready"

# ============================================================
# STEP 2: Install Ollama
# ============================================================
Write-Step "Checking Ollama..."

$OllamaPath = "C:\Users\VH\AppData\Local\Programs\Ollama"
$OllamaExe = Join-Path $OllamaPath "ollama.exe"

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Success "Ollama already installed"
} elseif (Test-Path $OllamaExe) {
    # Add to PATH for this session
    $env:Path = "$OllamaPath;$env:Path"
    Write-Success "Ollama found at $OllamaPath"
} else {
    Write-Warn "Installing Ollama..."
    $Installer = "$env:TEMP\ollama-setup.exe"
    try {
        Write-Host "  Downloading..." -NoNewline
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $Installer -UseBasicParsing
        Write-Host " Done" -ForegroundColor Green
        Write-Host "  Running installer..."
        Start-Process -FilePath $Installer -Wait
        $env:Path = "$OllamaPath;$env:Path"
        Write-Success "Ollama installed"
    } catch {
        Write-Fail "Failed: $_"
        Write-Host "Install manually: https://ollama.com"
        exit 1
    }
}

# Ensure Ollama is running
Write-Host "  Starting Ollama service..." -NoNewline
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    Write-Host "Running" -ForegroundColor Green
} catch {
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    Write-Success "Ollama started"
}

# ============================================================
# STEP 3: Download GGUF Model from Google Drive
# ============================================================
Write-Step "Checking GGUF model..."

$GGUFModel = Join-Path $ModelsDir "exam_r1_7b_q4.gguf"

if ((Test-Path $GGUFModel) -and ((Get-Item $GGUFModel).Length -gt 1GB)) {
    $size = [math]::Round((Get-Item $GGUFModel).Length / 1GB, 2)
    Write-Success "GGUF model exists ($size GB)"
} else {
    Write-Warn "Downloading GGUF model from Google Drive (~4.4 GB)..."
    Write-Host "  This may take 5-15 minutes depending on your connection" -ForegroundColor Gray
    
    # Remove corrupt/partial file if exists
    if (Test-Path $GGUFModel) { Remove-Item $GGUFModel -Force }
    
    $FileId = "1YiPfVjYLqPRwxfDWXmQVIdlUqFVn_nPh"
    $ConfirmUrl = "https://drive.google.com/uc?export=download&id=$FileId"
    
    # Use pip to install gdown for reliable large file downloads
    Write-Host "  Installing download tool..." -NoNewline
    pip install gdown --quiet 2>&1 | Out-Null
    Write-Host " Done" -ForegroundColor Green
    
    # Download with gdown (handles large file confirmation)
    $GdownPath = "$env:APPDATA\Python\Python313\Scripts\gdown.exe"
    if (-not (Test-Path $GdownPath)) {
        $GdownPath = "gdown"
    }
    
    Write-Host "  Downloading model..." -ForegroundColor Cyan
    
    $DownloadJob = Start-Job -ScriptBlock {
        param($gdown, $url, $output)
        & $gdown --fuzzy $url -O $output 2>&1
    } -ArgumentList $GdownPath, $ConfirmUrl, $GGUFModel
    
    # Monitor progress
    $startTime = Get-Date
    while ($DownloadJob.State -eq 'Running') {
        Start-Sleep -Seconds 5
        if (Test-Path $GGUFModel) {
            $size = [math]::Round((Get-Item $GGUFModel).Length / 1MB, 0)
            $pct = [math]::Min([math]::Round($size / 4466 * 100, 0), 100)
            $elapsed = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 0)
            Write-Progress -msg "Downloaded ${size}MB" -pct $pct
        }
    }
    
    $Result = Receive-Job -Job $DownloadJob
    Remove-Job -Job $DownloadJob
    
    if ((Test-Path $GGUFModel) -and ((Get-Item $GGUFModel).Length -gt 1GB)) {
        $size = [math]::Round((Get-Item $GGUFModel).Length / 1GB, 2)
        Write-Success "Model downloaded ($size GB)"
    } else {
        Write-Fail "Download failed!"
        Write-Host "Manual download: https://drive.google.com/file/d/1YiPfVjYLqPRwxfDWXmQVIdlUqFVn_nPh/view"
        Write-Host "Save as: $GGUFModel"
        exit 1
    }
}

# ============================================================
# STEP 4: Create Ollama Model
# ============================================================
Write-Step "Creating Ollama model..."

$Modelfile = Join-Path $ProjectRoot "Modelfile"

# Verify Modelfile points to correct GGUF
$ModelfileContent = Get-Content $Modelfile -Raw -Encoding UTF8
if ($ModelfileContent -notmatch "exam_r1_7b_q4\.gguf") {
    # Update Modelfile
    $NewContent = $ModelfileContent -replace "FROM\s+[^\n]+", "FROM ./models/exam_r1_7b_q4.gguf"
    $NewContent | Out-File -FilePath $Modelfile -Encoding UTF8
}

# Check if model already exists
$ExistingModels = & ollama list 2>&1
if ($ExistingModels -match "exam-r1-solver") {
    Write-Success "Model 'exam-r1-solver' already exists"
} else {
    Write-Host "  Creating model (this takes ~30 seconds)..."
    & ollama create exam-r1-solver -f $Modelfile 2>&1 | ForEach-Object {
        if ($_ -match "success") { Write-Success "Model created" }
    }
}

# ============================================================
# STEP 5: Pull Vision Model (optional)
# ============================================================
if (-not $SkipVision) {
    Write-Step "Checking vision model..."
    
    $ExistingModels = & ollama list 2>&1
    if ($ExistingModels -match "qwen2.5vl") {
        Write-Success "Vision model already exists"
    } else {
        Write-Warn "Downloading vision model qwen2.5vl:3b (~2 GB)..."
        & ollama pull qwen2.5vl:3b 2>&1 | ForEach-Object {
            if ($_ -match "success") { Write-Success "Vision model ready" }
        }
    }
}

# ============================================================
# STEP 6: Configure OpenCode Integration
# ============================================================
Write-Step "Configuring OpenCode integration..."

# Create .env file
$EnvFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $EnvFile)) {
    @'
OLLAMA_HOST=http://localhost:11434
CODE_MODEL=exam-r1-solver
'@ | Out-File -FilePath $EnvFile -Encoding UTF8
    Write-Success ".env file created"
}

# Create OpenCode startup script
$StartScript = Join-Path $ProjectRoot "start_opencode.ps1"
@'
# NLP Exam Solver - OpenCode Launcher
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OllamaPath = "C:\Users\VH\AppData\Local\Programs\Ollama"
$env:Path = "$OllamaPath;$env:Path"
$env:OLLAMA_HOST = "http://localhost:11434"
$env:CODE_MODEL = "exam-r1-solver"

Write-Host "Starting Ollama..." -NoNewline
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Write-Host " Ready" -ForegroundColor Green

Write-Host "`nOpenCode Commands:" -ForegroundColor Cyan
Write-Host "  /shell python exam_solver.py `"Your question`"" -ForegroundColor White
Write-Host "  /shell python exam_solver_pipeline.py image.jpg" -ForegroundColor White
Write-Host "  /shell python exam_solver.py --interactive" -ForegroundColor White

if (Get-Command opencode -ErrorAction SilentlyContinue) {
    Start-Process -FilePath "opencode" -WorkingDirectory $ProjectRoot
}
'@ | Out-File -FilePath $StartScript -Encoding UTF8
Write-Success "OpenCode launcher created: start_opencode.ps1"

# ============================================================
# STEP 7: Install Python Dependencies
# ============================================================
Write-Step "Installing Python dependencies..."

if (Get-Command python -ErrorAction SilentlyContinue) {
    $PyVer = python --version 2>&1
    Write-Host "  Python: $PyVer"
    
    pip install requests python-dotenv Pillow --quiet 2>&1 | Out-Null
    Write-Success "Dependencies installed"
} else {
    Write-Warn "Python not found. Install from https://www.python.org/"
}

# ============================================================
# STEP 8: Test Everything
# ============================================================
Write-Step "Running tests..."

# Test 1: Ollama API
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 5
    Write-Success "Ollama API responding"
} catch {
    Write-Fail "Ollama API not responding"
}

# Test 2: Model exists
$Models = & ollama list 2>&1
if ($Models -match "exam-r1-solver") {
    Write-Success "Model 'exam-r1-solver' available"
} else {
    Write-Fail "Model not found in Ollama"
}

# Test 3: Quick generation (run as background job with timeout)
Write-Host "  Testing generation (model loads on first use, ~60-90s)..." -NoNewline

$GenJob = Start-Job -ScriptBlock {
    $Body = @{
        model = "exam-r1-solver"
        prompt = "Write hello world in Python"
        stream = $false
        options = @{ num_predict = 16; num_ctx = 128 }
    } | ConvertTo-Json
    
    $Response = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body $Body -ContentType "application/json" -TimeoutSec 180
    return $Response.response
}

# Wait up to 120 seconds for generation
$GenCompleted = $GenJob | Wait-Job -Timeout 120
if ($GenCompleted) {
    $GenResult = Receive-Job -Job $GenJob
    if ($GenResult) {
        Write-Host "OK" -ForegroundColor Green
    } else {
        Write-Host "Empty response" -ForegroundColor Yellow
    }
} else {
    Write-Host "Loading (model warms up on first use)" -ForegroundColor Yellow
}
Remove-Job -Job $GenJob -Force

# Test 4: Python tests
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "  Running unit tests..." -NoNewline
    $TestResult = python -m pytest tests/ -q 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "All tests passed"
    } else {
        Write-Warn "Some tests failed (check output above)"
    }
}

# Test 2: Model exists
$Models = & ollama list 2>&1
if ($Models -match "exam-r1-solver") {
    Write-Success "Model 'exam-r1-solver' available"
} else {
    Write-Fail "Model not found in Ollama"
}

# Test 3: Quick generation (run as background job with timeout)
Write-Host "  Testing generation (model loads on first use, ~60-90s)..." -NoNewline

$GenJob = Start-Job -ScriptBlock {
    $Body = @{
        model = "exam-r1-solver"
        prompt = "Write hello world in Python"
        stream = $false
        options = @{ num_predict = 16; num_ctx = 128 }
    } | ConvertTo-Json
    
    $Response = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body $Body -ContentType "application/json" -TimeoutSec 180
    return $Response.response
}

# Wait up to 120 seconds for generation
$GenCompleted = $GenJob | Wait-Job -Timeout 120
if ($GenCompleted) {
    $GenResult = Receive-Job -Job $GenJob
    if ($GenResult) {
        Write-Host "OK" -ForegroundColor Green
    } else {
        Write-Host "Empty response" -ForegroundColor Yellow
    }
} else {
    Write-Host "Loading (model warms up on first use)" -ForegroundColor Yellow
}
Remove-Job -Job $GenJob -Force

# Test 4: Python tests
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "  Running unit tests..." -NoNewline
    $TestResult = python -m pytest tests/ -q 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "All tests passed"
    } else {
        Write-Warn "Some tests failed (check output above)"
    }
}

# ============================================================
# DONE!
# ============================================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  Installation Complete!" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Quick Start:" -ForegroundColor Yellow
Write-Host "  1. Start: .\start_opencode.ps1" -ForegroundColor White
Write-Host "  2. Or: ollama serve" -ForegroundColor White
Write-Host "  3. Then: python exam_solver.py `"Your question`"" -ForegroundColor White
Write-Host ""
Write-Host "OpenCode Usage:" -ForegroundColor Yellow
Write-Host "  /shell python exam_solver.py `"Tokenize text with NLTK`"" -ForegroundColor White
Write-Host "  /shell python exam_solver_pipeline.py exam.jpg" -ForegroundColor White
Write-Host ""
