# TS-GNN HPC -> Local Sync (PowerShell version)
# Usage: .\hpc\sync_down.ps1

param(
    [string]$Key = "$env:USERPROFILE\.ssh\id_bocconi",
    [string]$Remote = "3393519@slogin.hpc.unibocconi.it",
    [string]$RemoteDir = "~/ts-gnn"
)

$SshOpts = "-i `"$Key`" -o StrictHostKeyChecking=accept-new"

Write-Host ">>> Downloading results from $Remote..." -ForegroundColor Cyan

# Create local directories if they don't exist
if (!(Test-Path "results")) { New-Item -ItemType Directory "results" }
if (!(Test-Path "checkpoints")) { New-Item -ItemType Directory "checkpoints" }
if (!(Test-Path "logs")) { New-Item -ItemType Directory "logs" }

# SCP results
Invoke-Expression "scp $SshOpts -r ${Remote}:${RemoteDir}/results/* ./results/"
Invoke-Expression "scp $SshOpts -r ${Remote}:${RemoteDir}/checkpoints/* ./checkpoints/"
Invoke-Expression "scp $SshOpts -r ${Remote}:${RemoteDir}/logs/* ./logs/"

Write-Host ">>> Results downloaded." -ForegroundColor Green
