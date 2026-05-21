# TS-GNN Local (G: Drive) -> HPC Sync
# Usage: .\hpc\sync_up.ps1

param(
    [string]$Key = "$env:USERPROFILE\.ssh\id_bocconi",
    [string]$Remote = "3393519@slogin.hpc.unibocconi.it",
    [string]$RemoteDir = "~/ts-gnn",
    [switch]$IncludeRawData,
    [switch]$IncludeProcessedData,
    [switch]$IncludeExternalData
)

$ZipFile = "tsgnn_sync.zip"
$SshOpts = "-i `"$Key`" -o StrictHostKeyChecking=accept-new"
$DataArgs = @()
if ($IncludeRawData) { $DataArgs += "raw" }
if ($IncludeProcessedData) { $DataArgs += "processed" }
if ($IncludeExternalData) { $DataArgs += "external" }

Write-Host ">>> Zipping code via Python..." -ForegroundColor Cyan
if ($DataArgs.Count -gt 0) {
    python hpc/zip_code.py --include-data $DataArgs
} else {
    python hpc/zip_code.py
}

Write-Host ">>> Uploading to $Remote..." -ForegroundColor Cyan
# Crea la cartella sull'HPC se non esiste
Invoke-Expression "ssh $SshOpts $Remote `"mkdir -p $RemoteDir`""
Invoke-Expression "scp $SshOpts $ZipFile ${Remote}:${RemoteDir}/"

Write-Host ">>> Unpacking on HPC..." -ForegroundColor Cyan
Invoke-Expression "ssh $SshOpts $Remote `"cd $RemoteDir && unzip -q -o $ZipFile && rm $ZipFile && mkdir -p logs results checkpoints`""

Write-Host ">>> Sync complete." -ForegroundColor Green
