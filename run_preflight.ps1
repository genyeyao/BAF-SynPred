param(
    [string]$EnvironmentName = "bafsynpred",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectRoot
try {
    & conda run --no-capture-output -n $EnvironmentName python scripts/verify_release.py
    & conda run --no-capture-output -n $EnvironmentName python scripts/validate.py
    & conda run --no-capture-output -n $EnvironmentName python -m unittest discover -s tests -v
    & conda run --no-capture-output -n $EnvironmentName python scripts/smoke_test.py --device $Device
} finally {
    Pop-Location
}

