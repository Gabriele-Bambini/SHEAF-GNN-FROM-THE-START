# ssh_setup.ps1 — One-time: generate Windows SSH key and install it on HPC.
# Run from PowerShell as yourself (no admin needed).
# Usage: .\hpc\ssh_setup.ps1

$KeyPath    = "$env:USERPROFILE\.ssh\id_bocconi"
$HpcUser    = "3393519"
$HpcHost    = "slogin.hpc.unibocconi.it"
$HpcTarget  = "${HpcUser}@${HpcHost}"

# 1. Generate key (no passphrase — needed for unattended scripts)
if (Test-Path $KeyPath) {
    Write-Host "[ssh_setup] Key already exists at $KeyPath — skipping keygen."
} else {
    Write-Host "[ssh_setup] Generating ed25519 key (no passphrase for automation)..."
    ssh-keygen -t ed25519 -f $KeyPath -N "" -C "bocconi-hpc-automation"
    Write-Host "[ssh_setup] Key created."
}

# 2. Show the public key and push it to HPC authorized_keys
$PubKey = Get-Content "$KeyPath.pub"
Write-Host "`n[ssh_setup] Your public key:"
Write-Host $PubKey
Write-Host ""

Write-Host "[ssh_setup] Installing public key on HPC (password prompt will appear)..."
# Use ssh to append the key — password auth still works at this point
$cmd = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$PubKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo 'OK: key installed'"
ssh "${HpcTarget}" $cmd

# 3. Test passwordless login
Write-Host "`n[ssh_setup] Testing passwordless login..."
$test = ssh -i $KeyPath -o BatchMode=yes -o ConnectTimeout=10 $HpcTarget "echo SUCCESS"
if ($test -eq "SUCCESS") {
    Write-Host "[ssh_setup] Passwordless SSH works!"
    Write-Host "`nNow you can run:"
    Write-Host "  .\hpc\run_pipeline.ps1"
} else {
    Write-Warning "[ssh_setup] Test failed. Check ~/.ssh/authorized_keys on HPC."
}
