# CRC-GAD — one-time GitHub publish (run from CRC_GAD folder in PowerShell)
# Requires: Git for Windows (https://git-scm.com/download/win)
#           GitHub account + repo https://github.com/Amaanullakhan/CRC_GAD (create empty repo first if needed)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: git not found. Install Git for Windows, reopen PowerShell, then re-run this script."
    exit 1
}

if (-not (Test-Path .git)) {
    git init
}

git add .
git status

$msg = @"
Rebuild CRC-GAD v1.0-rebuild — reproducible experiments and paper tables.

- 6 datasets x 5 seeds in results/main_results.csv
- Auto-generated paper/generated/*.tex and paper/figures/*.png
- Simplified CoLA-inspired NumPy scorer + split conformal calibration
"@

git commit -m $msg 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Nothing to commit or commit failed — check git status"
}

git tag -f v1.0-rebuild

$remote = git remote get-url origin 2>$null
if (-not $remote) {
    git remote add origin https://github.com/Amaanullakhan/CRC_GAD.git
}

Write-Host ""
Write-Host "Next: git branch -M main"
Write-Host "       git push -u origin main"
Write-Host "       git push origin v1.0-rebuild"
Write-Host ""
Write-Host "Paper URL: https://github.com/Amaanullakhan/CRC_GAD/tree/v1.0-rebuild"
