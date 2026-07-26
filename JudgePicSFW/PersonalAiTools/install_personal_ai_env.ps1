param(
    [string]$RootFolder = "",
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [string]$TorchCudaIndex = "https://download.pytorch.org/whl/cu126",
    [bool]$PreferCudaTorch = $true
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-EnvProgress {
    param(
        [int]$Current,
        [int]$Total,
        [string]$Step,
        [string]$Detail
    )

    $payload = [ordered]@{
        current = $Current
        total = $Total
        step = $Step
        detail = $Detail
    } | ConvertTo-Json -Compress

    Write-Host "ENV_PROGRESS $payload"
}

if ([string]::IsNullOrWhiteSpace($RootFolder)) {
    $RootFolder = Join-Path $PSScriptRoot "..\Data\PersonalAi"
}

$totalSteps = 6
$RootFolder = [System.IO.Path]::GetFullPath($RootFolder)
$venvFolder = Join-Path $RootFolder ".venv"
$pythonExe = Join-Path $venvFolder "Scripts\python.exe"
$requirements = Join-Path $PSScriptRoot "requirements.txt"
$readyMarker = Join-Path $RootFolder ".personal-ai-env-ready"

Write-EnvProgress 1 $totalSteps "Prepare folders" "Preparing personal AI folder: $RootFolder"
New-Item -ItemType Directory -Force -Path $RootFolder | Out-Null

if (-not (Test-Path $pythonExe)) {
    Write-EnvProgress 1 $totalSteps "Create Python venv" "Creating local Python venv: $venvFolder"
    python -m venv $venvFolder
}

Write-EnvProgress 2 $totalSteps "Upgrade pip" "Upgrading pip through mirror."
& $pythonExe -m pip install --upgrade pip -i $Mirror --no-cache-dir --progress-bar off --timeout 30 --retries 2

$gpuDetected = $false
try {
    $gpuDetected = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match 'NVIDIA' }).Count -gt 0
}
catch {
    $gpuDetected = $false
}

$preferredCudaInstall = $false
$installMode = "cpu"
$torchCudaVersion = ""

if ($PreferCudaTorch -and $gpuDetected) {
    try {
        Write-EnvProgress 3 $totalSteps "Install PyTorch CUDA" "NVIDIA GPU detected. Installing CUDA PyTorch."
        & $pythonExe -m pip install --progress-bar off --timeout 60 --retries 3 torch torchvision --index-url $TorchCudaIndex --no-cache-dir
        $torchCudaVersion = (& $pythonExe -c 'import torch; print(torch.version.cuda or "")' | Select-Object -Last 1).Trim()
        if (-not [string]::IsNullOrWhiteSpace($torchCudaVersion)) {
            $preferredCudaInstall = $true
            $installMode = "cuda"
            Write-Host "Installed CUDA torch/torchvision."
        }
        else {
            throw "Installed torch is not CUDA-enabled."
        }
    }
    catch {
        Write-Host "CUDA torch install failed. Falling back to CPU build. $($_.Exception.Message)"
        & $pythonExe -m pip uninstall -y torch torchvision | Out-Null
        Write-EnvProgress 3 $totalSteps "Install PyTorch CPU" "CUDA install failed. Installing CPU PyTorch through mirror."
        & $pythonExe -m pip install --progress-bar off --timeout 30 --retries 2 torch torchvision -i $Mirror --no-cache-dir
    }
}
else {
    Write-EnvProgress 3 $totalSteps "Install PyTorch CPU" "CUDA preference disabled or NVIDIA GPU not found. Installing CPU PyTorch."
    & $pythonExe -m pip install --progress-bar off --timeout 30 --retries 2 torch torchvision -i $Mirror --no-cache-dir
}

Write-EnvProgress 4 $totalSteps "Install training dependencies" "Installing transformers, peft, pillow, and safetensors."
& $pythonExe -m pip install --progress-bar off --timeout 30 --retries 2 -r $requirements -i $Mirror --no-cache-dir

Write-EnvProgress 5 $totalSteps "Detect training device" "Checking PyTorch runtime device."
$torchInfo = (& $pythonExe -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' | Select-Object -Last 1).Trim()
$torchCudaVersion = (& $pythonExe -c 'import torch; print(torch.version.cuda or "")' | Select-Object -Last 1).Trim()
$installMode = if ($preferredCudaInstall -and -not [string]::IsNullOrWhiteSpace($torchCudaVersion)) { "cuda" } else { "cpu" }
$readyText = @(
    "installedAt=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "gpuDetected=$gpuDetected"
    "preferredCudaInstall=$preferredCudaInstall"
    "installMode=$installMode"
    "torchCudaVersion=$torchCudaVersion"
    "runtimeDevice=$torchInfo"
) -join [Environment]::NewLine

Set-Content -Path $readyMarker -Value $readyText -Encoding UTF8
Write-EnvProgress 6 $totalSteps "Environment ready" "Personal AI environment is ready: Python=$pythonExe, device=$torchInfo."
Write-Host "Personal AI environment ready: Python=$pythonExe, installMode=$installMode, runtimeDevice=$torchInfo"
