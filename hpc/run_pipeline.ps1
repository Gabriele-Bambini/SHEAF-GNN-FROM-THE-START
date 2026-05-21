# run_pipeline.ps1 - Full pipeline: sync -> sbatch -> poll -> pull results.
# Usage:
#   .\hpc\run_pipeline.ps1
#   .\hpc\run_pipeline.ps1 -SkipSync -PollIntervalSec 120

param(
    [string]$Key             = "$env:USERPROFILE\.ssh\id_bocconi",
    [string]$Remote          = "3393519@slogin.hpc.unibocconi.it",
    [int]   $PollIntervalSec = 60,
    [string]$EntryPoint      = "scripts/run_nature.py",
    [string]$Config          = "configs/nature.yaml",
    [string]$ExtraArgs       = "",
    [switch]$IncludeRawData,
    [switch]$IncludeProcessedData,
    [switch]$IncludeExternalData,
    [switch]$SkipSync
)

$RemoteRepo = "/home/3393519/ts-gnn"
$ScriptDir  = $PSScriptRoot
$SshArgsBase = @("-i", $Key, "-o", "StrictHostKeyChecking=accept-new")

if (-not $IncludeRawData -and -not $IncludeProcessedData -and -not $IncludeExternalData) {
    $IncludeProcessedData = $true
    $IncludeExternalData = $true
}

function Invoke-HpcCommand {
    param([string]$Command)
    $output = & ssh @SshArgsBase $Remote $Command
    return ($output | Out-String).Trim()
}

if (-not $SkipSync) {
    Write-Host "`n[1/4] Syncing code to HPC..."
    & "$ScriptDir\sync_up.ps1" -Key $Key -Remote $Remote `
        -IncludeRawData:$IncludeRawData `
        -IncludeProcessedData:$IncludeProcessedData `
        -IncludeExternalData:$IncludeExternalData
} else {
    Write-Host "`n[1/4] Skipping sync (--SkipSync)"
}

Write-Host "`n[2/4] Submitting SLURM job..."
$submitCmd = "cd $RemoteRepo && TSGNN_ENTRYPOINT='$EntryPoint' TSGNN_CONFIG='$Config' TSGNN_EXTRA_ARGS='$ExtraArgs' TSGNN_CLUSTER=1 TSGNN_STRICT_SCIENCE=1 sbatch hpc/submit.sbatch"
$submitOut = Invoke-HpcCommand $submitCmd
Write-Host "  $submitOut"

if ($submitOut -match '(\d+)$') {
    $JobId = $Matches[1]
    Write-Host "  Job ID: $JobId"
} else {
    throw "Could not parse job ID from: $submitOut"
}

Write-Host "`n[3/4] Polling job $JobId every ${PollIntervalSec}s..."
$done = $false
while (-not $done) {
    Start-Sleep -Seconds $PollIntervalSec
    $status = Invoke-HpcCommand "squeue -j $JobId -h -o '%T' 2>/dev/null"
    $ts = Get-Date -Format "HH:mm:ss"
    if ($status) {
        Write-Host "  [$ts] $status"
        if ($status -notin @("PENDING", "RUNNING", "CONFIGURING", "COMPLETING")) {
            $done = $true
        }
    } else {
        Write-Host "  [$ts] No longer in queue - COMPLETED"
        $done = $true
    }
}

$exitCode = Invoke-HpcCommand "sacct -j $JobId -n -o ExitCode --noheader 2>/dev/null | head -1"
Write-Host "  Exit code: $exitCode"
if ($exitCode -and $exitCode -notmatch '^0') {
    Write-Warning "Job may have failed (exit code: $exitCode)."
    Write-Warning "Inspect remote log with: ssh $Remote 'cat $RemoteRepo/logs/tsgnn_${JobId}.err'"
}

Write-Host "`n[4/4] Pulling results..."
& "$ScriptDir\sync_down.ps1" -Key $Key -Remote $Remote

Write-Host "`n=== Pipeline complete ==="
