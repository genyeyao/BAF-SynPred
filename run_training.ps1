param(
    [string]$EnvironmentName = "bafsynpred",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [switch]$HighPriority
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Arguments = @(
    "scripts/train.py",
    "--data-root", (Join-Path $ProjectRoot "data"),
    "--output-root", (Join-Path $ProjectRoot "runs"),
    "--device", $Device
)
if ($HighPriority) {
    $Arguments += "--high-priority"
}

Push-Location $ProjectRoot
try {
    & conda run --no-capture-output -n $EnvironmentName python scripts/verify_release.py
    & conda run --no-capture-output -n $EnvironmentName python scripts/validate.py
    & conda run --no-capture-output -n $EnvironmentName python scripts/preprocess.py
    & conda run --no-capture-output -n $EnvironmentName python @Arguments
} finally {
    Pop-Location
}

