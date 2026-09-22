param(
    [string]$EnvironmentName = "bafsynpred"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentFile = Join-Path $ProjectRoot "environment.yml"
$RequirementsFile = Join-Path $ProjectRoot "requirements-pyg-cu116.txt"
$PyGWheelIndex = "https://data.pyg.org/whl/torch-1.13.0+cu116.html"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Install Miniconda or Anaconda and open a new PowerShell terminal."
}

$EnvironmentList = (& conda env list | Out-String)
if ($EnvironmentList -notmatch "(?m)^\s*$([regex]::Escape($EnvironmentName))\s") {
    & conda env create -n $EnvironmentName -f $EnvironmentFile
} else {
    & conda env update -n $EnvironmentName -f $EnvironmentFile --prune
}

& conda run -n $EnvironmentName python -m pip install --upgrade "pip<24.1"
& conda run -n $EnvironmentName python -m pip install `
    --find-links $PyGWheelIndex `
    -r $RequirementsFile
& conda run -n $EnvironmentName python -m pip install --no-deps -e $ProjectRoot

Write-Host "BAF-SynPred was installed in conda environment '$EnvironmentName'."
Write-Host "Next: .\run_preflight.ps1 -EnvironmentName $EnvironmentName"

