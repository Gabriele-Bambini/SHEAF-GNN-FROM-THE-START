param(
    [string]$HostAlias = "bocconi-stud"
)

# Install the local public SSH key on Bocconi HPC.
# This script contains no password. SSH will ask for the password interactively.

$ErrorActionPreference = "Stop"

$pub = Join-Path $env:USERPROFILE ".ssh\id_bocconi.pub"
if (!(Test-Path -LiteralPath $pub)) {
    throw "Missing public key: $pub"
}

Write-Host "Installing public key on Bocconi HPC for alias '$HostAlias'." -ForegroundColor Cyan
Write-Host "If SSH asks for a password, type it in this terminal. It will not be saved in this script." -ForegroundColor Yellow

$remoteCommand = "mkdir -p ~/.ssh; chmod 700 ~/.ssh; cat >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
Get-Content -LiteralPath $pub | ssh $HostAlias $remoteCommand

Write-Host ""
Write-Host "Testing non-interactive key login..." -ForegroundColor Cyan
ssh -o BatchMode=yes -o ConnectTimeout=10 $HostAlias "echo hpc_ok"

Write-Host ""
Write-Host "If you see hpc_ok above, Codex can submit the Slurm smoke job next." -ForegroundColor Green
Read-Host "Press Enter to close"
