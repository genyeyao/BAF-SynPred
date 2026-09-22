param(
    [string]$EnvironmentName = "bafsynpred",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [switch]$ForcePreprocess
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataRoot = Join-Path $ProjectRoot "data"
$WeightsRoot = Join-Path $ProjectRoot "weights\canonical_5fold"
$OutputRoot = Join-Path $ProjectRoot "runs"
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

Push-Location $ProjectRoot
try {
    & conda run --no-capture-output -n $EnvironmentName python scripts/verify_release.py
    & conda run --no-capture-output -n $EnvironmentName python scripts/validate.py
    if ($ForcePreprocess) {
        & conda run --no-capture-output -n $EnvironmentName python scripts/preprocess.py --force
    } else {
        & conda run --no-capture-output -n $EnvironmentName python scripts/preprocess.py
    }
    & conda run --no-capture-output -n $EnvironmentName python scripts/evaluate.py `
        --run-dir $WeightsRoot `
        --data-root $DataRoot `
        --device $Device `
        --output (Join-Path $OutputRoot "canonical_5fold_evaluation.json")
} finally {
    Pop-Location
}

